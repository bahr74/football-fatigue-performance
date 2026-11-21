# M12 report 8: Human review readiness (v1)

Status: plan and queue **proposal** ready. **Nothing has been reviewed:** no human decision exists yet
(`data/processed/m11-human-review-v1/` has not been created) and no item is ever marked reviewed automatically. Date 2026-11-21.

## 1. Prioritised queue (`python -m football_fatigue.m11_explorer review-plan` → `data/processed/m11-review-priority-v1/`)

Each item appears once, in its highest tier. There are at most 3 items per match and tier. Selection inside a tier uses a hash of
the item id, not model correctness, except where the tier is defined by model behaviour (tiers 1, 5 and 6). Every row has
`review_status = "unreviewed"`. **Built: 175 items from 32 matches** (`queue.parquet`, `plan.json`).

| Tier | Category | Definition | Selected / eligible |
|---|---|---|---|
| 1 | Severe contradictions | Phase 2 top 25: physics confidently contradicts the provider label | 25 / 25 |
| 2 | Clone-like | Independent samples in a clone-like pair-period (duplicated / co-moving traces) | 30 / 1,029 |
| 3 | ESTIMATED-dominated | Both players ESTIMATED in the gap and > 50 % of the context ESTIMATED | 30 / 9,624 |
| 4 | Position jumps | Jump-transfer risk near the event | 30 / 309 |
| 5 | Ambiguous crossings | Local crossing; the GRU within 0.15 of 0.5, or GRU and HGB disagree | 30 / 2,189 |
| 6 | Multi-player | Exploratory N×M events with N ≥ 3, every solver wrong on the complete variant | 30 / 301 |

Tiers 1 and 2 test **label validity** (is the provider identity itself right?). Tiers 3 and 4 test **input validity**. Tiers 5 and
6 sample the **hard cases** the benchmark depends on.

## 2. Measurable acceptance criteria for freezing dataset v2

**Statistical basis.** Wilson 95 % intervals. With 30 reviewed items per tier and 0 to 1 errors, the upper bound stays below
about 17 %; 60 items are needed to bound it below about 8 %.

| ID | Criterion | Threshold | Evidence |
|---|---|---|---|
| V1 | Provider identity label error, tiers 2 to 5 pooled | Upper 95 % bound ≤ 5 % (needs about 120 items with ≤ 2 errors) | Human `identity_continuity_plausible` |
| V2 | Severe contradictions explained | ≥ 80 % of tier 1 classified (true swap / provider artefact / physics wrong) | Human `provider_artifact_suspected` |
| V3 | Label-error rate ESTIMATED vs VISIBLE | Reported with CIs; if ESTIMATED error is higher, ESTIMATED-dominated events go to a separate stratum, not the headline | Tier 3 vs other tiers |
| V4 | Reviewer agreement | Two reviewers on ≥ 30 shared items; Cohen's κ ≥ 0.6 on identity continuity | Double review |
| V5 | Shortcut controls in place | Label-free texture rule and timing-oracle baselines reported; headline on both-VISIBLE (two-player) and async variants (N×M) | Reports 2, 6 |
| V6 | Split hygiene | Match-disjoint folds frozen before any v2 model is trained; held-out matches untouched | Guard tests |
| V7 | Provenance | Every item labelled `synthetic` / `provider-derived` / `human-reviewed`; only items with a saved human decision may be called reviewed | Schema |
| V8 | Balance | No match contributes > 10 % of any stratum; per-N counts ≥ 200 events for N = 2 to 4 | Quality report (report 5) |

**Until V1, V2 and V4 hold, "provider-derived" labels must not be called ground truth.**

## 3. Compact reviewer guide

1. **Open the Studio.** Run `scripts\m11_studio.cmd`, then go to the Review queue workspace. Enter your reviewer name, which is
   required: decisions are append-only per reviewer.
   - **Current limit:** the Studio's queue is the existing 125 events (100 Phase 2 stratified events plus the 25 severe
     contradictions, which are tier 1 here).
   - Tiers 2 to 6 exist only as `queue.parquet`; this milestone allowed no dashboard change. Loading them into the Studio queue is
     a small follow-up that needs approval.
   - Tier 6 scenarios can be opened today in the Multi-player workspace by uploading their JSON.
2. **For each item, look at:**
   - the two (or N) trajectories ±2 s around the event;
   - the ESTIMATED shading;
   - the hidden-path reveal;
   - the speed panel.
3. **Answer the six questions.** Prefer `unsure` to guessing.
   - Positions plausible?
   - Players actually close (geometry only)?
   - Identity continuous, or a swap suspected?
   - Which provider artefact (smooth ESTIMATED interpolation, clone-like trace, jump, swap)?
   - An unexplained physical contradiction?
   - Is visual occlusion checkable? Always `no` without video.
4. **Do not infer occlusion.** Proximity is geometry; ESTIMATED means the provider filled the position, not that the player was
   seen or hidden.
5. **Do not look at model predictions** before answering the identity question (the queue does not show them; tier membership is
   known, which is unavoidable).
6. **Write a note** for every `no` or swap answer: what you saw, and when (seconds).
7. **Exports** are new timestamped files; nothing is overwritten. Double-review items are assigned by the coordinator, not chosen by
   the reviewer.

## 4. What is still missing

- Video for any event: visual occlusion cannot be checked. Every answer is about tracking-data plausibility only.
- A second reviewer for the agreement criterion V4.
- The N×M review view: tier 6 items are scenarios from the exploratory prototype. The Studio shows them through the existing
  multi-player page, but there is no dedicated question set for N×M links yet. A proposal: per incoming track, "which outgoing
  track is the same player, or none, or unsure".
