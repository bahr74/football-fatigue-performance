# M11 Step 2: real-first player identity matching benchmark (design note and pilot)

Status: 2026-11-06. Pilot on match 3812 only, using the 2 m threshold. Not done in this step: FOOTBALLTrace, model training,
synthetic swaps, and processing other matches.

## 1. Corrected objective

**Primary goal: identity preservation and recovery through close encounters.** Swap detection is not the goal.

- **The task:** given the incoming trajectories of the players in an encounter and the *anonymous* outgoing tracklets after it,
  decide which outgoing tracklet belongs to each incoming player.
- **The decision:** MATCHED, UNMATCHED or UNCERTAIN, with a cost, a margin and evidence.
- **Swaps are one failure mode only.** Synthetic swap injection stays a possible later stress test and is not part of this step.
- **Inputs are positional only:** numerical RAW tracking. Image appearance is not used.

## 2. What the data provides, and what masking changes

**The original data (unchanged):**
- RAW PFF positions at 29.97 fps, with about 15 Hz real updates;
- visibility, confidence and timestamps;
- provider identity labels (side, jersey).

Step 1 adds a non-destructive quality mask and pairwise proximity encounters (`m11-encounters-v2`).

**Identity masking (task A, the primary pilot):**
- **Merged into events:** pairwise 2 m encounters that share a participant (a pair member, or a third player within 2 m) and
  overlap or lie less than 1 s apart.
- **Ambiguity interval:** first to last close frame. It *keeps* its RAW observations, but as unlabelled point sets that are
  shuffled again in every frame. Uninterrupted provider slot continuity is therefore not visible.
- **Before window (2 s):** incoming tracklets, anonymous `in-k`, in seeded random order.
- **After window (2 s):** outgoing tracklets, anonymous `out-k`, in an independent seeded random order.
- **Time** is relative to the start of the interval.

**Hidden from the matcher:**
- jersey, player, team, side and slot;
- frame numbers, period, match ID and absolute time;
- the reference mapping;
- Step 1 categories and quality-mask bits derived from slot continuity.

**What masking does not change:** the geometry, timing, visibility and confidence of every observation. Nothing is interpolated,
removed, corrected or smoothed.

**Separation:**
- `root.instances.jsonl` is the only matcher-visible file. Its schema is a fixed whitelist (`INSTANCE_KEYS`, `TRACKLET_KEYS`,
  `DURING_KEYS`).
- `root.references.jsonl` and `root.eligibility.jsonl` are evaluation-only.
- Baselines receive the instances re-parsed from the written bytes and nothing else.
- Tests check:
  - the key whitelist and forbidden words;
  - relative time;
  - that the seeded order is unrelated to slot order;
  - that predictions are identical whatever the instance ID and after a JSON round trip.

**Independence:**
- Events are taken in time order.
- An event is rejected (`overlaps_earlier_instance`) if a participant's window overlaps one already used by an earlier instance.
  No observation serves two instances.
- Episodes longer than 6 s (sustained marking) are rejected as `ambiguity_too_long`.

## 3. Eligibility and tiers (transparent, fixed before results)

**Rejected:**
- more than 6 participants;
- an ambiguity interval over 6 s;
- substitution window, over-full side, unresolved attribution or ambiguous raw values;
- fewer than 2 incoming tracklets, or no outgoing tracklet;
- a window overlapping an earlier instance.

**Tiers:**

| Tier | Rule |
|---|---|
| `natural_suspicious_unverified` | A step-1 `DISCONTINUITY_CANDIDATE` in any participant's window. The reference itself is in doubt |
| `high_confidence_reference` | Every participant has incoming and outgoing tracklets. Both windows at least 90% observed. ESTIMATED share at most 0.1. No implausible jump. No frame-wide shift. Not cut by a segment edge |
| `lower_confidence_reference` | Everything else accepted, with the reasons recorded |

A tracklet exists when its window holds at least 0.3 s of available samples. A stationary player (held positions) is therefore
valid.

**Difficulty** is recorded from geometry only and is never used for selection:
- crossing;
- heading change over 45°;
- more than 2 players;
- an interval over 3 s.

## 4. Metrics (evaluation.py)

**Per incoming tracklet, five outcomes:**
- `correct_match`;
- `wrong_match` (including any match when the player has no outgoing tracklet);
- `correct_unmatched` (genuinely absent);
- `missed_unmatched`;
- `uncertain`.

**Rates over incoming tracklets:**
- correct match rate;
- wrong match rate;
- unmatched rate;
- uncertain rate;
- abstention rate;
- coverage.

**Also reported:**
- false-association rate = wrong matches / MATCHED decisions;
- complete assignment accuracy, per instance;
- conflicts (two incoming tracklets matched to one outgoing tracklet).

**Breakdowns:** tier, number of players, difficulty, team relation, crossing, absent tracklets, and whether the event contains a
step-1 trusted encounter.

All of these are **agreement with the provider reference**. Precision, recall, F1, IDF1 and HOTA are not reported.

## 5. Pilot results, match 3812, τ = 2 m (`data/processed/m11-matching-v2/3812/`)

**Eligibility:** 3,157 encounters gave 1,274 events: 845 accepted and 429 rejected.

| Accepted tier | Instances |
|---|---|
| high confidence | 214 |
| lower confidence | 628 (ESTIMATED share 597, implausible jump 182) |
| natural suspicious | 3 |

| Rejected (primary reason) | Events |
|---|---|
| ambiguity too long | 191 |
| overlaps an earlier instance | 183 |
| more than 6 participants | 47 |
| side over-count | 4 |
| no outgoing tracklet | 2 |
| substitution window | 1 |
| fewer than 2 incoming tracklets | 1 |

**Step-1 trusted encounters (342):**

| Outcome | Encounters |
|---|---|
| high-confidence tier | 185 |
| lower-confidence tier | 4 |
| rejected | 153 (merged into crowds 81, overlap 37, too long 35) |

**Instances:** 1,799 incoming tracklets and 1,799 outgoing tracklets.

| | Count |
|---|---|
| 2-player instances | 761 |
| 3-player instances | 63 |
| 4-player instances | 17 |
| 5-player instances | 4 |
| challenging | 687 |
| crossing | 346 |
| sharp turn | 455 |

Median interval: 2.0 s.

**High-confidence tier, provider-reference agreement:**

| Baseline | Correct | Wrong | Uncertain | Unmatched | False assoc. | Complete |
|---|---|---|---|---|---|---|
| nearest | 0.428 | 0.344 | 0.150 | 0.077 | 0.445 | 0.042 |
| constant_velocity | 0.437 | 0.339 | 0.185 | 0.039 | 0.437 | 0.108 |
| hungarian_cv | 0.365 | 0.255 | 0.326 | 0.055 | 0.412 | 0.341 |
| chain_during | 0.934 | 0.000 | 0.066 | 0.000 | 0.000 | 0.925 |

The three gap-style baselines see only the before and after windows, and they are close to chance. In 2-player high-tier
instances the true pairing has the lower summed endpoint distance in 66.8% of cases, and in only 53.7% with constant-velocity
extrapolation (205 instances).

`chain_during` agrees with the reference almost perfectly whenever it commits. Over all tiers it made 4 wrong matches out of
1,799, all in the lower tier.

## 6. What can and cannot be claimed

**Can be claimed:**
- the benchmark construction is reproducible, non-destructive and leakage-tested;
- the eligibility audit;
- that naive motion baselines without encounter-interval observations agree with provider identity barely better than chance
  over about 2 s intervals.

**Cannot be claimed:**
- verified identity accuracy;
- occlusion (proximity is not occlusion);
- generality beyond match 3812.

**Task A is close to tautological for continuity-based matchers.**
- RAW positions follow the provider's own tracker. Chaining the unlabelled interval points mostly re-derives the provider's
  linkage, and the high tier was *selected* for no jumps.
- Task A therefore measures agreement with provider continuity, not the ability to recover identity from a genuine gap.
- The meaningful difficulty sits in task B (withholding the interval).

**Evaluation biases:**
- Selection uses slot-continuity flags, so it favours cases the provider tracked smoothly.
- Time-order de-duplication drops later contacts.
- The ESTIMATED filter removes 597 events, which skews the high tier towards visible players.
- 81 trusted encounters were lost to crowd merging.
- Per-sample visibility and confidence are exposed as quality evidence, but confidence patterns could act as a weak identity cue.
- RAW float values could in principle be looked up in the raw file. Only an adversarial matcher could exploit this, and it is
  unavoidable with RAW geometry.

**Untuned parameters:** gates, margins, the 1 s merge, the 6 s cap and the 0.3 s tracklet minimum were all set a priori and
never tuned.

## 7. How this prepares FOOTBALLTrace E0

- FOOTBALLTrace corrects tails after an occlusion candidate. Its decision can be evaluated with exactly these instances and
  outcomes: incoming → outgoing assignment, with false corrections counted as wrong matches against the reference.
- Every method shares the same inputs and protocol, alongside the baselines.
- Next, task B (a withheld interval) on the same instances is the right stress test for FOOTBALLTrace-style trajectory
  reasoning.
- Expansion to other development matches reuses the same code (a step-1 raw pass first).

## 8. Files and commands

- **Code:** `src/football_fatigue/matching/` (`benchmark.py`, `baselines.py`, `evaluation.py`, `__main__.py`).
- **Tests:** `tests/test_matching.py` (17).
- **Run:** `.\.venv\Scripts\python.exe -m football_fatigue.matching build --root data/raw --match 3812`. It refuses
  non-development matches and never overwrites.
- **Outputs (`data/processed/m11-matching-v2/3812/root.*`):**
  - config, eligibility, instances (matcher-visible), references (evaluation-only), predictions, summary, manifest;
  - the manifest records the source checksums, the code sha256, timing and memory.
- **`m11-matching-v1/3812`:** a superseded first pilot (4 s chaining merged man-marking into 233 s events). Kept unchanged.
