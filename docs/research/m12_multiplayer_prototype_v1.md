# M12 report 5: Exploratory multi-player (N×M) dataset prototype (v1)

Status: **EXPLORATORY, not frozen, not dataset v2.** Date 2026-11-21.

- Code: `src/football_fatigue/m11_explorer/nxm_dataset.py`, run with
  `scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-dataset` (CPU, 411 s).
- Output: `data/processed/m11-nxm-exploratory-v1/{scenarios.jsonl.gz, index.parquet, build.json, manifest.json}`.
- Quality report: `data/processed/m11-nxm-solver-comparison-v1/quality.json`.

## Provenance and labelling

| Property | Value |
|---|---|
| Matches | The 32 train/CV development matches only (guarded by `C.checked`; held-out refused) |
| Source events | Interaction-catalogue link-0 groups at R = 1.0 m; centre = the group state with the largest connected component (2-player groups: closest approach of the edge) |
| Truth | Provider identities: **`provenance = "provider-derived"`, `human_reviewed = false`** on every scenario. No example is human-reviewed; none is synthetic |
| Hidden intervals | Synthetic observation gaps cut into provider tracks. **Proximity is geometry; nothing here is a verified occlusion** |
| Eligibility | Every member has ONE continuous provider chain over the whole window (identity continuous by construction). Contexts use only samples strictly outside each player's own hidden interval, resampled to 15 Hz without extrapolation |
| Outgoing ids | Anonymous, randomly permuted (`out-k`); the validator rejects identity keys on outgoing tracks |

## Variants (per event, seeded; variants of one event are augmentations, not independent samples)

| Variant | Construction | Covers |
|---|---|---|
| `complete` | N×N, synchronous 0.8 s gap | Baseline |
| `async_var` | Base gap 0.6 to 1.6 s, per-player ±25 % jitter, ±0.4 s centre offsets, 20 % context dropout | Asynchronous events, variable gaps, partial visibility |
| `unequal` | `async_var`, one member's outgoing removed (null truth); a non-member within 6 m added as a new outgoing when available | Unmatched incoming, new outgoing, distractors |
| `extra_out` | `async_var` + the nearest eligible non-member's outgoing only: N×(N+1) | New outgoing / distractor without null |

## Size and coverage

**Totals.** 8,540 scenarios from 2,213 events in 32 matches. Events per match: minimum 54, median 68.5, maximum 83.

**Events by N.**

| N | Events |
|---|---|
| 2 | 611 |
| 3 | 939 |
| 4 | 404 |
| 5 | 259 |

**Scenarios by shape.**

| N×M | 2×1 | 2×2 | 2×3 | 3×2 | 3×3 | 3×4 | 4×3 | 4×4 | 4×5 | 5×5 | 5×6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Scenarios | 202 | 1,631 | 410 | 102 | 2,715 | 835 | **7** | 1,205 | 398 | 777 | 258 |

The catalogue has few large simultaneous groups: at R = 1.0 m, groups of 4 or more players are rare. Sizes 3 to 5 are therefore
almost exhaustively used, while size 2 is capped at 25 events per match. N > M with N ≥ 4 is nearly absent, because a 6 m
distractor is almost always available. That is a coverage gap.

## Quality report

| Aspect | Measurement | Reading |
|---|---|---|
| Null frequency | 25.9 % of scenarios contain a null; 8.2 % of incoming tracks are null | Only from `unequal` |
| Distractors | 3,803 scenarios with a new outgoing; distance p10 / p50 / p90 = 1.4 / 2.5 / 4.6 m | Realistic, near the group |
| ESTIMATED dependence | Mean context ESTIMATED share 0.70; 88 % of scenarios have a track that is mostly ESTIMATED; only 5.6 % are all visible | **As in the two-player data, provider interpolation dominates** |
| Label risk | 3.9 % of scenarios touch a label-risk edge; 12.4 % involve ESTIMATED co-motion | Provider labels not verified |
| Crossings | 44 % have a synchronous crossing | Hard, by design |
| Unrealistic motion | 6.3 % have a sample-to-sample speed > 12 m/s; p99 14.0 m/s | Provider jumps; candidate review items |
| Dependence | 2,213 events form 1,774 clusters (events sharing a player within 10 s) | CIs must be match-level (done) |
| Match balance | No match above 4 % of events | Balanced |
| `repeated_interaction` flag | 1.0 on every scenario | **Defective flag (`order_in_chain > 0` is always true for these groups); not used** |

## Shortcut probes (label-free rules)

| Probe | Result | Verdict |
|---|---|---|
| Timing oracle: assign by \|t_start(out) − t_end(in) − true base gap\| | Exact 0.22 complete, 0.45 async, 0.36 unequal, 0.14 extra_out | Per-player jitter prevents a timing shortcut. Async 0.45 is still above chance for small N; do not drop the jitter |
| Outgoing id order equals incoming order | 0.335 | Ids are permuted, so there is no order leak |
| Texture similarity rule | N = 2: 0.52 / 0.57 (ESTIMATED ≤ 50 % / > 50 %); N = 3: 0.20 / 0.24; N = 4: 0.05 / 0.10; N = 5: 0.02 / 0.04 | Close to chance (0.5 / 0.17 / 0.04 / 0.008); the texture signal is weak at the N×M level but nonzero, higher when ESTIMATED |
| Count shortcut | P(null \| fewer outgoing than incoming) = 1.0; otherwise 0.23 | **Structural by construction**: a null is certain when M < N. Solvers must not be credited for it; the null-recall figures below are dominated by such cases |

## What this prototype is not

- It is not visual. It is not human-reviewed and not frozen.
- It is not representative of real tracker failures. Members are selected for continuous provider chains, as in Stage 3, so the
  same selection bias applies.
- It must not be used to claim video performance.
