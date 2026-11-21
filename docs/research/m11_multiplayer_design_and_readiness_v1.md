# Multi-player (N × M) identity association: design, readiness and algorithm options (v1)

**Date:** 2026-11-20 · **Code:** `src/football_fatigue/m11_explorer/multiplayer.py` · **Tests:** `tests/test_m11_studio.py` (N × M section)
**Status:** schema, synthetic curriculum, deterministic reference solver and metrics are **implemented and verified on synthetic data only**.
No multi-player dataset has been frozen and no learned N × M model exists.

## 1. Problem

N incoming tracklets with known identity end before an observation gap; M ≥ 0 anonymous outgoing tracklets start after it.
Output: a partial one-to-one assignment incoming → outgoing **or null** (player left / not re-observed), plus unmatched outgoing
(new player / distractor). The binary MATCHED / SWAPPED models answer only the 2 × 2 case with exactly one valid permutation pair.

## 2. Schema `m11-multiplayer-association-v1`

```json
{"schema": "m11-multiplayer-association-v1", "synthetic_example": true, "level": 3,
 "incoming":  [{"id": "in-0", "label": "player A", "t": [...], "x": [...], "y": [...]}],
 "outgoing":  [{"id": "out-7f3a", "t": [...], "x": [...], "y": [...]}],
 "truth": {"in-0": "out-7f3a", "in-1": null}}
```

`validate()` enforces: incoming ids unique; outgoing **anonymous** (only `id,t,x,y`; identity-bearing keys refused); equal-length,
time-ordered arrays; truth one-to-one, keys = incoming ids, values = outgoing ids or null. `public()` strips truth;
the solver runs on the public copy (tested: identical decision with and without truth).

## 3. Curriculum (synthetic generator `generate(level, seed)`)

| Level | Content | Status |
|---|---|---|
| L1 | 2 × 2, synchronous gap | ✔ synthetic |
| L2 | 3 × 3, synchronous | ✔ synthetic |
| L3 | 3 × 2 (one player not re-observed → null) | ✔ synthetic |
| L4 | 4 × 4, asynchronous start/end offsets | ✔ synthetic |
| L5 | 3 × 3 + distractor outgoing + null | ✔ synthetic |
| L6 | real group, provider truth, human-checked | designed only (needs review + dataset v2 gates) |
| L7 | 5 × 5 mixed: null, distractor, asynchrony, partial visibility | ✔ synthetic |
| L8 | real video tracklets (Session B) | pending real video |

## 4. Reference solver (deterministic, no learned parameters)

Kalman forward prediction (σ_a 8 m/s², σ_m 0.05 m, same as Phase 2) from each incoming end to each outgoing start → Mahalanobis d²;
infeasible if the outgoing starts before the incoming ends or d² > χ²₂(0.99) = 9.21; Hungarian assignment on an augmented matrix
with a null column per incoming at the gate cost (`use_null=False` forces a full assignment). ~10 ms per scenario.

**Synthetic benchmark** (`python -m football_fatigue.m11_explorer nxm-benchmark --n 300` → `data/processed/m11-nxm-synthetic-benchmark-v1/`) [V]:

| Level | Exact assignment | Link P / R | Null recall | ID switches / scenario |
|---|---|---|---|---|
| L1 | 0.987 | 0.987 / 0.987 | – | 0.03 |
| L2 | 0.960 | 0.960 / 0.960 | – | 0.08 |
| L3 | 0.597 | 0.732 / 0.864 | **0.00** | 0.23 |
| L4 | 0.890 | 0.926 / 0.926 | – | 0.22 |
| L5 | 0.437 | 0.761 / 0.814 | **0.00** | 0.79 |
| L7 | 0.337 | 0.729 / 0.784 | 0.10 (precision 1.0) | 0.80 |

**Findings.** (1) Synthetic L1 (0.987) is far easier than the real 0.5 m / 0.8 s population (Kalman forward 0.667 BA): synthetic motion
is smooth; these numbers test code, not football. (2) The null option almost never fires: a nearby distractor or the other players'
outgoing tracks usually stay inside the 0.99 gate, so the solver links instead of declaring "not re-observed". A gate is not a
calibrated "left the scene" model. (3) Errors grow with N and with distractors, as expected.

## 5. Real groups (exploratory only)

`from_real_group(tracks, slots, period, t_centre, gap, async, seed)` cuts a gap into real provider tracks of a link-0 simultaneous
group chosen in the explorer. Truth = provider identities (**not human-verified**); groups can be ESTIMATED-heavy or clone-like.
Not a dataset, not frozen, not reported as a result.

## 6. Dataset v2 / multiplayer v1 gates (from `m11_dataset_v2_design.md`)

Required before freezing: human review of the 125-item queue (provider-truth error rate, clone-like rate), approval of the v2 scope
(radius set, group sizes, asynchrony), and a held-out-safe split reusing the frozen 32-match partition. **None is met** →
only synthetic schema tests and this non-frozen exploratory path are allowed.

## 7. Algorithm options (ordered)

1. **Kalman + Hungarian-null** (implemented; reference).
2. Better null handling: per-track "exit" prior from pitch boundary / gap length; calibrated on development matches only.
3. Learned **link scorer**: pairwise GRU/HGB score per (incoming, outgoing) link, fed into the same Hungarian — reuses the binary
   models' encoders; needs N × M training data (gated by dataset v2).
4. **Group-aware** models (set/attention encoder over all tracks; joint assignment): only after (3) shows a gain; GPU-gated.
5. Appearance / jersey cues: only with real video (Session B), not from tracking data.

Metrics for all: exact assignment, link precision/recall, null precision/recall, identity switches, per-level and per-match bootstrap.
