# M13 report 4: Human validation readiness (v1)

Status: **proposal and tooling only. Nothing has been reviewed.**
- `data/processed/m11-human-review-v1/` does not exist; `review.decisions()` returns 0 rows (checked 2026-11-21).
- No item is, or will be, marked reviewed by code.
- Provider-derived identities are **not** manually verified ground truth.

Date 2026-11-21.

- Code: `src/football_fatigue/m11_explorer/review_v2.py`, run with
  `scripts\session_a_python.cmd -m football_fatigue.m11_explorer review-export`.
- Output: `data/processed/m11-review-priority-v2/` (write-once).
- `m11-review-priority-v1` is unchanged (tested against its manifest).

## 1. Queue v2: what changed and why

- **Tiers 1 to 5 are carried over unchanged** from `m11-review-priority-v1`: same item ids and selection.
- **M12 tier 6 is dropped.** It was drawn from the v1 N×M prototype, which had duplicate encounters and the timing shortcut (report 1).
  It is replaced by two N×M tiers built from v2 canonical encounters, with **one item per encounter**:

| Tier | Category | Definition | Purpose |
|---|---|---|---|
| 1 | Severe contradictions | Phase 2 top 25: physics confidently contradicts the provider label | Label validity |
| 2 | Clone-like | Duplicated / co-moving traces | Label validity |
| 3 | ESTIMATED-dominated | Both players ESTIMATED in the gap and > 50 % of context | Input validity |
| 4 | Position jumps | Jump-transfer risk near the event | Input validity |
| 5 | Ambiguous crossings | Local crossing; GRU near 0.5, or GRU and HGB disagree | Hard two-player cases |
| 6 | N×M hard (v2) | N ≥ 3, `async` variant; both L2 HGB and S2 wrong on the untouched test fold | Hard multi-player cases: does the provider identity hold where motion fails? |
| 7 | N×M null / new (v2) | `null_new`, N ≥ 3, L2 HGB wrong | Is a "player left" or a "new player" decision plausible? |

**Selection rules.**
- At most 3 items per match and tier.
- Order is set by a hash of the item id. Model correctness is used only where the tier is defined by it (tiers 1, 5, 6, 7).
- Selection never uses provider labels to choose "interesting" answers.

**Double review.** 5 items per tier, chosen by hash, are flagged `double_review = True` for agreement (about 35 items).

The tier counts actually built are in section 7.

## 2. Labels: exact meaning

**Two-player items** (existing `review.QUESTIONS`, unchanged):

| Field | Allowed answers | Meaning |
|---|---|---|
| `positions_plausible` | unsure / yes / no | Both trajectories are physically possible (speed, acceleration, no teleport) around the event |
| `players_actually_close` | unsure / yes / no | Geometric proximity in provider coordinates only. **Not occlusion** |
| `identity_continuity_plausible` | unsure / yes / no - swap suspected / no - other break | The provider identity of each player is continuous through the event. "Swap" = the labels of the two players appear exchanged |
| `provider_artifact_suspected` | none / smooth ESTIMATED interpolation / clone-like duplicated trace / position jump / identity swap / other | Which artefact, if any, explains what is seen |
| `unexplained_contradiction` | unsure / no / yes | A physical contradiction that no flag explains |
| `visual_occlusion_checkable` | no (tracking data only) / yes (video available) | Always "no" in this project today |

**N×M items** (`plan.json → question_nxm`; answered in `review_template.csv`):
- **Per incoming track:** which outgoing track is the same player, "none - the player did not reappear in the window", or
  "unsure". Recorded in `nxm_links` as `in-0=out-xxxx; in-1=none; in-2=unsure`.
- **Per scenario:** `positions_plausible` and `provider_identity_continuous` (unsure / yes / no - swap suspected).
  `visual_occlusion_checkable` is always "no".
- **Blinding:** the reviewer opens `nxm_items/<id>.json`, which has the provider answer key removed. The key is in `nxm_key.json`
  (coordinator only). A reviewer who saw the key must mark the item `unsure`.

**ESTIMATED is not verified visibility.** ESTIMATED means the provider filled the position. It is never evidence that the player
was seen, nor that the player was hidden.

## 3. Reviewer instructions

1. Enter a reviewer name. Decisions are append-only per reviewer (`review.save_decision`); nothing is overwritten.
2. Look only at the tracking material:
   - trajectories ±2 s;
   - ESTIMATED shading;
   - the speed panel;
   - for N×M items, the scenario JSON in the Multi-player workspace (upload).
   Do not look at model predictions or the answer key.
3. Prefer `unsure` to guessing.
4. Write a note for every `no`, swap or `none` answer: what was seen, and at what time.
5. Double-review items are assigned by the coordinator. The second reviewer must not see the first reviewer's answers.

**Ambiguous or unverifiable.** Mark `unsure`, never `yes`, when any of these holds:
- either player's track is ESTIMATED through the whole window;
- the players move in parallel within 0.5 m (clone-like);
- a jump larger than about 3 m within one sample occurs inside the window;
- the outgoing candidates are indistinguishable by motion.

Unsure items are counted and reported. They are **excluded** from the error-rate numerator *and* denominator, and their rate is
reported separately; a high unsure rate is itself a finding.

## 4. Statistics: label-error confidence intervals and agreement

**Error rate.** Error = a reviewed (non-unsure) item answered `no - swap suspected` or `no - other break` for identity continuity.
It is reported with an **exact Clopper-Pearson 95 %** interval; Wilson intervals are shown alongside.
`review_v2.clopper_pearson` / `wilson` are tested.

| Errors among n reviewed | Exact 95 % CI |
|---|---|
| 0 / 30 | [0, 11.6 %] |
| 1 / 30 | [0.1 %, 17.2 %] |
| 0 / 60 | [0, 6.0 %] |
| 2 / 120 | [0.2 %, 5.9 %] |
| 5 / 120 | [1.4 %, 9.5 %] |

**Sample size for an upper bound ≤ 5 %:**

| Errors allowed | Items needed |
|---|---|
| 0 | 72 |
| 1 | 110 |
| 2 | 142 |
| 3 | 173 |

For an upper bound ≤ 10 %: 36 / 54 / 70 items. *Correction to M12 report 8:* "about 120 items with ≤ 2 errors" came from a
Wilson approximation; the exact requirement is 142.

**Agreement.** On the double-review subset, report:
- raw agreement;
- the confusion table;
- **Cohen's κ** on identity continuity (target ≥ 0.6; `review_v2.cohen_kappa`, tested).

Disagreements go to a third reviewer. The adjudication is stored as a separate reviewer record. Original answers are never edited.

**Stratification.** Error rates are reported per tier, and for ESTIMATED-dominated vs other items, each with its own interval.
Pooled rates are labelled with their tier mix.

## 5. Acceptance gates for promoting exploratory data toward a frozen dataset

The M12 criteria V1 to V8 stay in force, with V1 corrected and N×M gates added.

| ID | Gate | Threshold | Status |
|---|---|---|---|
| V1 | Provider identity error, tiers 2 to 5 pooled | Exact upper 95 % ≤ 5 % (≥ 142 non-unsure items if 2 errors) | **Not met** (0 decisions) |
| V2 | Severe contradictions explained | ≥ 80 % of tier 1 classified | Not met |
| V3 | ESTIMATED vs other error rates | Reported with CIs; separate stratum if higher | Not met |
| V4 | Agreement | ≥ 30 double-reviewed items, κ ≥ 0.6 | Not met (no second reviewer) |
| V5 | Shortcut controls | Shortcut-only baselines reported; headlines stratified by visibility | **Met for N×M v2** (report 2) |
| V6 | Split hygiene | Match-disjoint folds and calibration partitions, held-out untouched | **Met** (tested) |
| V7 | Provenance | provider-derived / human-reviewed separated | **Met** (schema) |
| V8 | Balance | No match > 10 % of a stratum; ≥ 200 events for each N = 2 to 4 | **Met for v2:** N = 2: 800, N = 3: 1,982, N = 4: 334 encounters |
| V9 (new) | N×M link error | Exact upper 95 % ≤ 10 % on ≥ 45 reviewed tier 6 and 7 encounters | Not met |
| V10 (new) | Rare cells declared | Every reported stratum has ≥ 30 encounters from ≥ 10 matches, or is marked "too rare" | Met (report 1: N ≥ 4 all-visible = 6 encounters, too rare) |

**Until V1, V2, V4 and V9 hold, no N×M or two-player label may be called ground truth, and dataset v2 must not be frozen.**

## 6. Inspect and export

| File | Use |
|---|---|
| `queue.parquet` / `queue.csv` | The full queue, `review_status = unreviewed`, with a `double_review` flag |
| `review_template.csv` | One row per item, with blank columns for every question plus `nxm_links`, `reviewer` and `notes`. Offline review is possible; the coordinator imports it through `review.save_decision`, so decisions stay append-only |
| `nxm_items/*.json` | Blinded N×M scenarios (schema `m11-multiplayer-association-v1`, no truth), loadable in the Studio Multi-player page |
| `nxm_key.json` | Coordinator-only provider answer key |
| `plan.json` | Tiers, counts, question sets, statistics plan, number of existing decisions (0) |

**Not done in this milestone (no dashboard changes allowed).** The Studio review workspace still lists only the 125 original
items (tiers 1 to 5 overlap them partly). Loading `m11-review-priority-v2` into the Studio needs approval (A4).

## 7. Built queue (`review-export`, 2026-11-21)

| Tier | Selected | Eligible | Double review |
|---|---|---|---|
| 1 Severe contradictions | 25 | 25 | 5 |
| 2 Clone-like | 30 | carried over | 5 |
| 3 ESTIMATED-dominated | 30 | carried over | 5 |
| 4 Position jumps | 30 | carried over | 5 |
| 5 Ambiguous crossings | 30 | carried over | 5 |
| 6 N×M hard (v2) | 30 | 983 encounters | 5 |
| 7 N×M null / new (v2) | 15 | 1,801 encounters | 5 |
| **Total** | **190** from 32 matches | | **35** |

- M12 tier 6 (30 items, v1 prototype) was dropped.
- All 190 items are `unreviewed`.
- `human_decisions_existing = 0` and the decision store does not exist.
- The 45 blinded N×M items contain no `truth`, slot, player-id or hidden-interval fields (checked).
- The double-review subset (35) meets the V4 minimum of 30 items once a second reviewer answers it.
- Meeting V1 needs at least 72 to 142 non-unsure items from tiers 2 to 5 (120 are queued), so the queue may need extension after
  the first round, depending on the error and unsure rates.
