# M11 Stage 1: three-state methods, identity-free candidates and label tiers (pilot on 3812 and 3816)

Status: **implemented and run on the two authorised development matches only. Stopped for review.** No validation or test match was
read. No model was trained. No other development match was processed.

Output: `data/processed/m11-window-local-methods-pilot-v4` (manifest SHA-256 prefix `8331965bc644eb92`).
Code: `src/football_fatigue/window_intersections/local.py` (states, causes, resumable runner) and `candidates.py` (new).
Tests: `tests/test_window_candidates.py` (new, 16), `tests/test_window_local.py` (1 test updated for h = 0.10).

## What was built

| Item | Implementation |
|---|---|
| Three states | `method_states`: int8 0 FAIL / 1 PASS / 2 UNDETERMINED for every F1 to F4 configuration. F1 is never undetermined. F2 is PASS when the observed minimum is ≤ R, because the unobserved part can only lower it. With the guard on a truncated neighbourhood it is UNDETERMINED, because the true arg-min may lie outside. F2 is FAIL only when fully observed. F3 is FAIL if F1 fails, else the F2 state. F4 is PASS if the gap is ≤ τ, because the gap can only shrink. F4 is UNDETERMINED only when an occupancy end *facing the other player* is capped. |
| Causes with side | `{arr,pt}_h*_trunc_side` (before / after / both) and `_trunc_cause` (`a_chain_start`, `b_chain_end`, …, `no_common_coverage`). `occ_*_facing_cap_cause` (`a_end_search_limit`, `b_start_chain_start`, …). |
| Motion | Chord speed (±0.2 s) at each arrival and its maximum within ±1 s. `implausible_motion` is set when the chord speed exceeds 12 m/s (configurable). |
| Memory-safe and resumable | One match at a time, written to `matches/<m>.partial/` with atomic writes. `match_done.json` (SHA-256 of each file) is written last, then the directory is renamed. `--resume` re-hashes completed matches, discards partial ones and refuses a changed config or code. |
| Independent trajectory checks | Knot tracks are compared with `core.positions_at` (linear: random times; PCHIP: interior knots). Every native vertex is compared with the raw Step 1 sample. |
| Identity-free candidates | `candidates.py`: exact episodes with d(t) ≤ 0.50 m per pair and chain overlap. Episodes ≤ 1 s apart are merged into one encounter, and the candidate is the minimum of that encounter. Crossing metadata and F1 to F4 links are attached after selection and never filter. |
| Label tiers | HIGH or LOWER for each gap (0.4, 0.8, 1.6 s), with an ordered funnel and reasons (below). |

## Acceptance results

**Tests.**
- Focused: 33 passed.
- Full repository suite (pipefail, exit 0): **974 passed, 6 skipped**, run twice (before and after the check fix below).

**Agreement with pilot-v2.**
- Common columns are identical (atol 1e-12) for both matches and both variants: 67 descriptor columns and all 207 mask columns.
- The only v2 column absent from v4 is `arr_h0p25_time_within_r_s`. It moved to the new primary h (`arr_h0p10_…`).
- All v1 and pilot-v2 inputs were hash-checked against their manifests.

**Dense and trajectory checks (400 events per match and variant).**
- Distance violation: 0.
- Occupancy end-point error: ≤ 1.0 ms.
- Curve versus `positions_at`: ≤ 7e-15 m (linear) and ≤ 1e-11 m (PCHIP).
- Raw vertices: 0.0 m error on 378,784 vertices.

**Raw data.** `raw_unchanged = true`.

**Crash and resume.** pilot-v3 was run in four steps:
1. A simulated crash (`--simulate-crash-after 1`, abrupt `os._exit(3)`) after 3812.
2. `--resume` was started, then hard-killed (`Stop-Process -Force`) during 3816. This left `3816.partial` holding only the linear files.
3. A further `--resume` verified 3812, which was byte-unchanged, discarded the partial directory and recomputed 3816.
4. pilot-v4 then ran straight through. Its **14 per-match parquet files are byte-identical to v3's.**

**Runtime and memory.**
- Per match: about 119 s (3812) and 103 s (3816). Of that, linear descriptors take about 30 s, PCHIP about 40 s and candidates 25–36 s.
- Figures: 127 s.
- Total: 352 s.
- Peak working set: 1.77 GB, reached at the figure and aggregation stage. RSS after each match was 0.81–0.87 GB.
- Estimate for 40 matches: about 75 min of per-match work.

**Why pilot-v3 is superseded by v4 (kept, not deleted).**
- v3's new PCHIP curve check reported a false 4.4 mm failure. The check compared the two *clipped end points* of a knot track, which are linear interpolations on the polyline, with the cubic. It now compares interior knots only, and a PCHIP test was added.
- The data are unaffected (byte-identical).
- An earlier attempt failed on a column-read bug before writing any data. It was renamed to `m11-window-local-methods-pilot-v3-aborted-attempt1`, which contains an empty partial directory. It was not deleted.

## Three-state counts (linear, 23,482 primary events)

| Configuration | PASS | UNDETERMINED | FAIL |
|---|---:|---:|---:|
| F1 τ0.30 | 1,511 | 0 | 21,971 |
| F1 τ0.20 | 956 | 0 | 22,526 |
| F2 arr h0.10 R0.25 | 1,605 | 3,035 | 18,842 |
| **F2 arr h0.10 R0.25 + guard** (primary) | **1,416** | 3,106 | 18,960 |
| F2 pt h0.10 R0.25 | 1,259 | 3,053 | 19,170 |
| F3 τ0.30 arr (no guard) | 684 | 61 | 22,737 |
| **F3 τ0.30 arr + guard** (primary) | **653** | 92 | 22,737 |
| F3 τ0.20 arr + guard | 540 | 46 | 22,896 |
| F4 ε0.05 / 0.10 / 0.25 | 1,699 / 1,933 / 2,878 | 29 / 57 / 191 | rest |
| F4 tier | 1,838 | 44 | 21,600 |

- PCHIP F2 + guard: 1,429 / 3,120. PCHIP F3 + guard: 662 / 93.
- **UNDETERMINED is large for F2 (13%).** Of the 3,106 truncated neighbourhoods:
  - 1,389 are truncated after the arrivals, 1,196 before, and 521 on both sides;
  - 226 have no common coverage at all;
  - causes are split evenly between the two players and between chain start and chain end.
- **Guard and truncation.** Compared with the boolean masks, the guard on a truncated neighbourhood moves 57 (F2) and 31 (F3) mask passes to UNDETERMINED. Pilot-v2 counted them as passes.
- **F4 facing caps are rare** (29–191). At ε0.25, 234 events have a facing cap: 60 come from the search limit and the rest from a chain end.
- **Implausible motion** (chord speed > 12 m/s within ±1 s) affects 888 events (3.8%).

## Identity-free candidates

**Population.** 2,328 candidates: 1,130 in 3812 and 1,198 in 3816.
- **Deduplication:** 3,554 episodes and 8,091 local minima form 2,328 encounters. 604 encounters merged several episodes, and 1,126 have more than one local minimum.
- **Strictness:** 1,168 candidates have d ≤ 0.25 m.
- **Team relation:** 1,531 opposite-team and 797 same-team.

**Crossing and non-crossing mixture (metadata only).** 1,467 local crossings, 436 window-only crossings and 425 non-crossing candidates.

| Candidate d_min | Local crossing | Window-only crossing | No crossing |
|---|---:|---:|---:|
| d ≤ 0.25 m | 935 | 136 | 97 |
| 0.25 < d ≤ 0.50 m | 532 | 300 | 328 |

**Coverage.**
- Every v1 event whose local minimum is ≤ 0.50 m (2,883) lies inside a candidate encounter with d_min no larger than its own.
- All 1,416 F2-primary and 653 F3-primary PASS events are covered.

**Cohort links (candidate level).**

| Cohort | PASS | UNDETERMINED | FAIL | No linked event |
|---|---:|---:|---:|---:|
| F2 primary | 941 | 144 | 934 | 309 |
| F3 primary | 627 | 42 | 1,350 | 309 |

**Leakage evidence.**
- Selection uses only d(t) = |A(t) − B(t)|, which depends on the *unordered* positions.
- Tested: exchanging the two players' identities after the instant they coincide turns a crossing into a bounce (the SWAPPED configuration). The selection columns stay identical and only the crossing metadata changes.
- Tested: relabelling the slots leaves the candidate set unchanged.
- The population is a mixture: 63% are local crossings and 37% are not.
- Crossing class and cohort columns are written after selection and never remove a candidate.

## Label-quality funnel (gap 0.8 s, sequential survivors)

| Condition | Survive | Fail independently |
|---|---:|---:|
| Window in period | 2,326 | 2 |
| Minimum's episode not truncated | 2,233 | 93 |
| One chain covering gap ± 0.5 s (continuous reference) | 2,056 | 271 |
| No duplicate / unattributed sample on the used chain | 2,056 | 0 |
| No identity-risk flag in the window | 2,051 | 18 |
| No jump / frame-wide-shift flag on the used chain | 2,051 | 0 |
| Chord speed ≤ 12 m/s | 2,006 | 86 |
| **≥ 0.5 s VISIBLE before and after, both players** | **395 HIGH** | 1,918 |

**Other gap lengths.** HIGH is 403 at gap 0.4 s and 384 at gap 1.6 s. 1,933 candidates are kept as LOWER, each with its reasons.

**HIGH by group (gap 0.8 s).**
- By crossing class: 258 local crossings, 70 window-only crossings and 67 non-crossing.
- By distance: 164 have d ≤ 0.25 m.
- By cohort: 137 F2-primary PASS and 112 F3-primary PASS.

**Two policy observations:**
1. **The VISIBLE requirement is the binding constraint.**
   - Only 43% of available samples in 3812 are VISIBLE.
   - Without the VISIBLE requirement, 2,006 candidates would be HIGH. A sensitivity count that leaves the policy unchanged gives 434 at ≥ 0.25 s and 466 at ≥ 0.10 s.
2. **Implausible-jump flags.** Chains are already cut at implausible jumps, so a jump flag sits on the first sample of the *next* chain.
   - When counted over the whole window, the flag removed 286 of the 1,130 candidates in 3812.
   - The flags are now evaluated only on the chain part that a sample would use. The chain's first sample is excluded, because a flag there marks the step *into* the chain. This removes 0.
   - Identity-risk flags are still evaluated over the full window, conservatively.

## Matched-control feasibility (HIGH only, same match, d band × slower chord-speed band)

| Cohort (HIGH local crossings) | Cases | With ≥ 1 control | Matchable 1:1 |
|---|---:|---:|---:|
| F2 primary | 128 | 93 | 10 |
| F3 primary | 111 | 79 | 9 |
| F1 τ0.30 | 187 | 153 | 43 |
| F4 tier | 199 | 165 | 46 |

There are only 67 HIGH non-crossing candidates across the two matches. Strict 1:1 matched controls within a match are not feasible at this size. Controls need pooling across matches, sampling with replacement, or coarser strata. That is a Stage 2 design decision.

## Blockers and open issues

1. **The HIGH yield is about 200 per match at the 0.8 s gap** (395 across 2 matches). Over 40 matches this extrapolates to about 7,900 HIGH candidates, about 1,300 of which are non-crossing. Most losses come from the VISIBLE rule. Whether to keep 0.5 s VISIBLE needs a decision.
2. The F2 UNDETERMINED share (13%) is dominated by chain ends near the arrivals. Under the current policy these events are excluded from the PASS cohort and are not treated as FAIL.
3. The held-out freeze was **not done**. It is proposed below for approval.
