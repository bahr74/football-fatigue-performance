# Session B M13: ByteTrack activation ablation (CPU, 2026-11-21)

**Verdict: NOT PROMOTED. T0 stays the production baseline.**

Lowering `track_activation_threshold` produces more tracks and more covered GT boxes. It does **not** improve identity continuity:
- GT-verified swaps rise by 9–25 %;
- false alerts rise;
- the HOTA change is match-dependent (−0.015 to +0.031).

The tune-selected cell gains +0.011 check HOTA, and almost all of that comes from the low-light match 117092.

## 1. Semantics (verified in supervision 0.30.9 `core.py`, unit-tested)

| quantity | rule | production (activation 0.5) | activation 0.3 |
|---|---|---|---|
| first association | conf ≥ activation | ≥ 0.5 | ≥ 0.3 |
| second association | 0.1 < conf < activation | 0.1–0.5 | 0.1–0.3 |
| **new track** | conf ≥ activation **+ 0.1** (`det_thresh`) | **≥ 0.6** | ≥ 0.4 |
| lost-track buffer | `int(fps / 30 × 75)` | 62 frames at 25 fps | same |

**Correction to M12:** M12 said boxes ≤ 0.5 cannot start a track. The real threshold is 0.6.

Because T0 feeds only boxes with conf > 0.4, every activation ≤ 0.3 gives T0 an identical tracker. So T0-A30 = T0-A20 exactly (confirmed: identical metrics).

## 2. Design

| item | value |
|---|---|
| inputs | T0 (cached boxes conf > 0.4, no turf filter) and T0L (conf > 0.1). Unchanged from M12 |
| varied | only `track_activation_threshold` ∈ {0.5 (production), 0.4, 0.3, 0.2}. Matching threshold, buffer, frame rate and detector are unchanged |
| declared cells | T0, T0L, T0-A03, T0L-A03 (factorial 2 × 2). The grid adds 0.4 and 0.2 |
| tune / check | 117093, 118575 / 117092, 118576 |
| selection (pre-declared, tune only) | max mean tune HOTA. Cells within 0.002 are broken by IDF1, then fewer IDSW |
| scoring | TrackEval through `detection.replay.score` (same code as M12). Engine v2 + eval-v1. pipeline-v1 decomposition |
| outputs | `_diag/activation-ablation-v1/`, `tracking/T0-A03-…`, `tracking/T0L-A03-…`, `_diag/pipeline-m13-v1/`, `_diag/engine-activation-v1/`. All new folders; nothing overwritten (baseline `verify()` = unchanged) |

Command: `python -m football_fatigue.video.failures activation --data-root F:/Football-Fatigue-Project/data`. The full chain is in the handoff.

## 3. Tune grid (mean of 117093 and 118575)

| cell | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | FP | FN | coverage | missing GT/frame | new tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | 0.4164 | 0.6795 | 0.2558 | 0.4128 | 0.801 | 614 | 2373 | 35,841 | 16,023 | 0.939 | 1.34 | 364 |
| T0-A40 | 0.4184 | 0.6758 | 0.2596 | 0.4223 | 0.795 | 627 | 2685 | 38,575 | 14,880 | 0.944 | 1.24 | 412 |
| T0-A30 (= A20) | 0.4216 | 0.6743 | 0.2642 | 0.4265 | 0.792 | 632 | 2696 | 39,224 | 14,838 | 0.944 | 1.24 | 472 |
| T0L | 0.4138 | 0.6749 | 0.2543 | 0.4138 | 0.792 | 608 | 1601 | 41,356 | 12,853 | 0.951 | 1.07 | 346 |
| T0L-A40 | 0.4195 | 0.6690 | 0.2639 | 0.4267 | 0.780 | 624 | 1681 | 45,296 | 12,014 | 0.954 | 1.00 | 397 |
| T0L-A30 | **0.4277** | 0.6614 | 0.2777 | 0.4441 | 0.766 | 621 | 1725 | 49,677 | 11,430 | 0.957 | 0.95 | 440 |
| T0L-A20 | 0.4265 | 0.6525 | 0.2797 | **0.4443** | 0.748 | 607 | 1824 | 54,851 | 10,902 | 0.959 | 0.91 | 476 |

**Selected: T0L-A20.** It is within 0.002 HOTA of T0L-A30 and wins the tie-break on IDF1 by 0.0002, which is a coin-flip margin.

The tune means hide opposite effects by match. T0L-A30 vs T0 is:
- 117093: HOTA 0.4234 → 0.4026 (**−0.021**);
- 118575: HOTA 0.4093 → 0.4528 (**+0.044**).

Runtime: 28–44 s per tracker run per match. Process peak memory: 249 MB.

## 4. Check (117092, 118576), scored once after selection

| cell | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | FP | FN | coverage | new tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | 0.3632 | 0.6552 | 0.2035 | 0.3631 | 0.775 | 859 | 5145 | 20,926 | 37,524 | 0.858 | 503 |
| T0L | 0.3681 | 0.6609 | 0.2074 | 0.3638 | 0.780 | 840 | 3232 | 25,555 | 31,543 | 0.880 | 465 |
| T0-A03 | 0.3585 | 0.6520 | 0.1988 | 0.3524 | 0.768 | 917 | 5966 | 24,995 | 35,212 | 0.867 | 666 |
| T0L-A03 | 0.3646 | 0.6446 | 0.2076 | 0.3597 | 0.749 | 903 | 3813 | 37,293 | 28,017 | 0.894 | 635 |
| **T0L-A20 (selected)** | **0.3741** | 0.6308 | 0.2232 | 0.3699 | 0.723 | **937** | 4143 | **45,341** | 26,625 | 0.899 | 700 |

Per match, HOTA of the selected cell vs T0:
- 117092 (low light): 0.3070 → 0.3380 (+0.031);
- 118576: 0.4195 → 0.4101 (**−0.009**).

The declared T0L-A03 cell is +0.0175 on 117092 and −0.015 on 118576.

**Selection bias.** The selected cell's tune gain (+0.010) roughly reappears on check (+0.011). But it reappears on a single match, and the other check match goes the other way. With two matches per split, neither number is a reliable expected gain.

## 5. Identity continuity vs "more tracks" (all 4 dev matches, 526,760 evaluable GT observations)

pipeline-m13-v1 decomposition (same rules as M12):

| profile | ok | untracked | ID changes | after a gap | **previous ID still output on another player** |
|---|---|---|---|---|---|
| T0 | 89.46 % | 0.80 % | 1,890 | 1,369 | **733** |
| T0L | 91.20 % | 2.07 % | 1,885 | 1,288 | 789 |
| T0-A03 | 90.11 % | 0.13 % | 1,991 | 1,379 | 796 |
| T0L-A03 | 92.11 % | 1.13 % | 2,041 | 1,283 | **914 (+25 %)** |

The "no box" share is 6.37 % in every profile: no tracker setting changes it.

Engine v2 (occlusion engine), check matches:

| profile | PO precision | PO recall | PO F1 | false occlusion alerts | missed occlusion episodes |
|---|---|---|---|---|---|
| T0 | 0.659 | 0.704 | 0.680 | 248 | 238 |
| T0L | 0.650 | 0.715 | 0.680 | 262 | 230 |
| T0-A03 | 0.629 | 0.743 | 0.679 | 325 | 207 |
| T0L-A03 | 0.610 | 0.764 | 0.676 | **358 (+44 %)** | 191 |

On tune, PO F1 rises from 0.693 (T0) to 0.724 (T0-A03) and 0.716 (T0L-A03). On check it stays flat.

**Reading:**
- A lower activation lets low-confidence boxes start tracks and match in the first round. That raises coverage, AssA and PO recall.
- It also starts tracks on fragments and on the neighbour's box. Evidence:
  - new tracks: +39 % on check for the selected cell;
  - FP: ×2.2;
  - MOTA: −0.05;
  - verified swaps: +9 % (T0-A03) to +25 % (T0L-A03);
  - false occlusion alerts: +31 to +44 %.
- Fewer detection-loss events in T0L-A03 come mainly from the T0L input, not from activation. T0-A03 has *more* detection-loss events than T0, because there are more short tracks to lose.

## 6. Decision

- **Not promoted.** Production defaults are untouched (`tracking.py` adds the two profiles but does not change `RECOMMENDED_TRACKER_INPUT`).
- The two profiles stay available for inspection in the player and the bench tab.
- A future promotion would need three things:
  1. more than two matches per split;
  2. a pre-declared identity constraint (verified swaps not increasing);
  3. a detector that does not starve low-light frames. Most of the gain sits in 117092, where the real problem is detection.
