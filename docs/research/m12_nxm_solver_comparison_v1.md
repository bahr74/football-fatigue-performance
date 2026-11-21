# M12 report 6: N×M association, deterministic solver variants (v1)

Status: **EXPLORATORY** (provider-derived truth on the exploratory prototype of report 5; not human-reviewed, not frozen, not
video). Date 2026-11-21.

- Code: `src/football_fatigue/m11_explorer/nxm_compare.py`, run with
  `scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-compare` (CPU, 850 s).
- Output: `data/processed/m11-nxm-solver-comparison-v1/{comparison.json, quality.json, synthetic_reference.json,
  per_scenario.parquet, manifest.json}`.
- Tests: `tests/test_m11_nxm.py`.

## Protocol

- **Cross-fitting:** scenarios are split by match into the frozen Stage 4 outer folds (8). Every tuned parameter is chosen on the
  other 7 folds; no threshold comes from a test fold. Tuning uses a bounded subset of about 1,500 training scenarios per fold.
- **Assignment:** all systems use Hungarian assignment, one-to-one. Inputs come from `multiplayer.public` (truth removed, tested).
- **Metrics:**
  - exact assignment (every incoming correct, including nulls);
  - link precision, recall and F1;
  - identity switches per scenario;
  - false-link rate;
  - null recall and precision.
- **Uncertainty:** match-level bootstrap (32 matches, 1,000 draws), on single systems and on paired differences.
- **Dependence:** about 4 scenarios per event; the match-level bootstrap covers it.

## Systems

| ID | Cost | Nulls |
|---|---|---|
| **S0** original | Forward Kalman Mahalanobis d2 (`multiplayer.solve`, σ_a default) | Gate χ²(2, 0.99) = 9.21 for both sides |
| **S1** improved motion | Bidirectional d2: mean of forward (incoming → outgoing start) and backward (outgoing filtered in reverse time → incoming end); time-order feasibility | Same gate |
| **S2** improved + tuned | S1 with σ_a ∈ {2, 4, 8, 16, 32} and separate incoming / outgoing null costs ∈ {0.5 … 36}, tuned per fold | Tuned |

**Equivalence check.** S0 reproduces `multiplayer.solve` exactly (60 real scenarios plus 45 synthetic; tested). S1 is symmetric
under time reversal (tested).

## Overall (8,540 scenarios, 32 matches)

| System | Exact [95 % CI] | Link P | Link R | Link F1 | ID switches / scenario | False-link rate | Null recall |
|---|---|---|---|---|---|---|---|
| S0 | 0.416 [0.404, 0.427] | 0.588 | 0.633 | 0.610 | 1.05 | 0.412 | 0.083 |
| S1 | 0.483 [0.472, 0.492] | 0.652 | 0.702 | 0.676 | 0.85 | 0.348 | 0.086 |
| S2 | 0.488 [0.475, 0.500] | 0.660 | 0.702 | 0.680 | 0.83 | 0.340 | 0.131 |

**Paired versus S0.**
- S1: **+0.067** [0.057, 0.075].
- S2: **+0.072** [0.061, 0.082].

S2 is not significantly better than S1. Its tuned parameters vary by fold: σ_a 2 to 8, incoming null 0.5 to 4, outgoing null 1 to
16. The incoming null cost hit the bottom of the grid (0.5) in 4 folds, which means the tuning prefers to leave incoming tracks
unmatched cheaply.

## Breakdowns (exact)

| Stratum | n | S0 | S1 | S2 |
|---|---|---|---|---|
| N = 2 | 2,243 | 0.576 | 0.612 | 0.620 |
| N = 3 | 3,652 | 0.382 | 0.451 | 0.450 |
| N = 4 | 1,610 | 0.330 | 0.423 | 0.433 |
| N = 5 | 1,035 | 0.326 | 0.408 | 0.421 |
| complete | 2,213 | 0.604 | 0.695 | 0.672 |
| async_var | 2,213 | 0.550 | 0.651 | 0.645 |
| extra_out (new outgoing) | 1,901 | 0.436 | 0.505 | 0.522 |
| **unequal (null + distractor)** | 2,213 | **0.078** | **0.083** | **0.117** |
| gap ≤ 0.7 s | 627 | 0.515 | 0.592 | 0.606 |
| gap 0.7 to 1.0 s | 4,179 | 0.509 | 0.594 | 0.591 |
| gap 1.0 to 1.3 s | 1,903 | 0.332 | 0.382 | 0.386 |
| gap > 1.3 s | 1,831 | 0.258 | 0.298 | 0.318 |
| context visible | 901 | 0.447 | 0.523 | 0.525 |
| context ESTIMATED ≤ 50 % | 1,678 | 0.299 | 0.372 | 0.363 |
| context ESTIMATED > 50 % | 5,961 | 0.445 | 0.508 | 0.517 |
| synchronous crossing | 3,757 | 0.323 | 0.393 | 0.399 |
| no crossing | 4,783 | 0.489 | 0.553 | 0.558 |

Shape detail (S0 / S1 / S2):

| Shape | S0 | S1 | S2 |
|---|---|---|---|
| 2×2 | 0.56 | 0.60 | 0.60 |
| 3×3 | 0.37 | 0.44 | 0.44 |
| 4×4 | 0.31 | 0.41 | 0.42 |
| 5×5 | 0.33 | 0.41 | 0.41 |
| 5×6 | 0.30 | 0.39 | 0.45 |

## Confidence and abstention (descriptive)

**Method.** Per-link margin = optimal total cost with that link forbidden minus the optimal total cost. The curve keeps the most
confident links; no threshold was tuned on test data.

| System | Link precision at 100 % coverage | 80 % | 60 % | 40 % |
|---|---|---|---|---|
| S1 | 0.652 | 0.692 | 0.717 | 0.730 |
| S2 | 0.660 | 0.695 | 0.709 | 0.719 |

The physics margin ranks links only weakly: precision stays below about 0.73 at any coverage. This is not a usable abstention
signal on its own (compare report 7).

## Reading

1. **Real N×M is much harder than the synthetic benchmark.** On synthetic levels, S0 exact is 0.985 (L1), 0.89 (3 players) and
   0.445 (4 to 5 players with distractor and null). On provider-derived real groups it is 0.60 even for the complete N×N variant
   and 0.33 for 4×4.
2. **Nulls are the main failure.** With a removed outgoing track plus a distractor, physics solvers are exact in only 8 to 12 % of
   scenarios: they link the distractor instead. Null recall (8 to 13 %) is almost entirely the structurally forced M < N cases.
3. **Bidirectional motion helps** (+0.067 exact, −0.2 identity switches per scenario). Tuning gating and null costs adds little.
4. **Longer gaps degrade strongly** (0.59 → 0.30 from 0.7–1.0 s to > 1.3 s).
5. **Physics also does better with ESTIMATED context** (0.52 vs 0.36 for partly visible). Provider interpolation makes motion more
   predictable; visible measured positions are noisier. The same caveat as report 2 applies.
