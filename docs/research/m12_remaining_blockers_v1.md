# M12 report 10: Remaining blockers and approvals needed (v1)

Date 2026-11-21.

## Blockers

| # | Blocker | Affects | What unblocks it |
|---|---|---|---|
| B1 | No GPU run under fair-benchmark v2: real s/epoch, best epochs and memory for the larger configurations are unknown | Fair GRU / LSTM comparison | **Approval A1**: GPU pilot (about 1.6 to 2.5 GPU-h [EST]) |
| B2 | Full fair-benchmark budget not approved (8 folds × 5 seeds, compute-matched halving, ≥ 2 distributions) | Final four-model comparison | Approval A2 after the pilot; sized from the pilot ledger |
| B3 | The historical two-player lead is partly a provider-texture effect; no texture-neutral headline exists | Any claim that the GRU "understands motion" | `robust_mix` / `visible_only` training plus both-VISIBLE headline (in A2) |
| B4 | No human-reviewed labels at all (0 decisions) | Dataset v2 freeze; any "ground truth" claim | Approval A3: reviewer time for the 175-item queue plus a second reviewer for 30 shared items |
| B5 | N×M prototype coverage: N > M with N ≥ 4 nearly absent (7 scenarios); `repeated_interaction` flag defective | N×M v2 design | Fix the flag; add `unequal` variants without distractor for N ≥ 4 (small code change, CPU) |
| B6 | Explicit null still weak (exact 0.37 on null + distractor) | Real deployment, where unmatched tracks are common | More null-structured training data; scenario-level null prior; dormant-track evidence |
| B7 | No calibrated video clip with independent identity truth | Any Session B integration | Agreement with Session B plus calibration with uncertainty (outside this repository) |
| B8 | Abstention thresholds and probabilities never fitted on a separate calibration partition | Deployable confidence | Calibration-partition protocol for N×M (CPU; needs v2 split) |
| B9 | Shared GPU with Session B | Scheduling of A1 / A2 | The pilot refuses to start while another compute process is on the GPU; coordinate a window |
| B10 | Studio review queue shows only the 125 historical items | Reviewing tiers 2 to 6 conveniently | Approval A4: small Studio change to load `m11-review-priority-v1` (this milestone forbade dashboard changes) |

## Approvals needed (none has been given)

| ID | Request | Cost | Command |
|---|---|---|---|
| A1 | GPU pilot, 7 configurations, fold 0, seed 0, cap 400 | About 1.6 to 2.5 GPU-h [EST], < 1.5 GB GPU | `scripts\session_a_python.cmd -m football_fatigue.m11_fair gpu-pilot --approval <id> --device cuda` |
| A2 | Fair benchmark v2 (after A1) | To be sized from the A1 ledger | `... m11_fair train --family <f> --dist <d> --search halving --approval <id> --device cuda` |
| A3 | Human review of the priority queue | Reviewer hours (175 items; about 30 double-reviewed) | Studio review workspace (tier 1 today), queue file for the rest |
| A4 | Studio: load the priority queue (tiers 2 to 6) and an N×M link question set | Small UI change | n/a |
| A5 | N×M prototype v1.1 fixes (B5) and a calibration-partition protocol (B8) | CPU, under an hour | n/a |
