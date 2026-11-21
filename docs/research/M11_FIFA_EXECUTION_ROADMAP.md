# M11: FIFA execution roadmap (15 phases)

Status: **proposal, nothing implemented**. Written 2026-11-13. Companion to `M11_FIFA_SCIENTIFIC_METHOD_DECISION.md`.

Runtimes are estimates **[I]** for this machine: 8 CPUs, GTX 1650 Ti 4 GB. The only measured basis is about 1 min per match for the proximity code at R ≤ 2 m.

Rules that hold in every phase:

- Never modify `data/raw/` or any existing versioned output. All new outputs go to new versioned directories with a manifest and source hashes.
- No FIFA validation or test access, and no SoccerTrack validation access, unless a phase says so *and* the user approves.
- Inputs are x, y and t only. Metadata is used only for labels, grouping and strata.
- No package installation. `sklearn` and `torch` are already available.
- Git is the user's responsibility.

---

## Phase 1: Protocol freeze and approval gate

| | |
|---|---|
| Inputs | Method decision doc, E1 results |
| Code to reuse | — |
| New | `PROTOCOL.md` (pre-registered thresholds, bands, tiers, gap rules, metrics, pass/fail criteria) |
| Outputs | `data/processed/m11-fifa-v4-protocol/PROTOCOL.md` |
| Checks | Every parameter in decision doc §6 has a value; every later phase's pass/fail rule is written down |
| Runtime | < 1 h |
| Dependencies | — |
| Pass / fail | Pass if the user and the advisor approve |
| Stop | Advisor rejects the "descriptor-only" status of crossing |
| Approval | **User + advisor** |

## Phase 2: 40-match data audit and quality-flag sidecar

| | |
|---|---|
| Inputs | Raw cache via `matching.benchmark.load_source`; Step 1 outputs |
| Code to reuse | `intersections.exact.trajectories`, `intersections.detect.duplicate_positions`, `encounters.quality` (ESTIMATED bit, `cadence`), `step1_verify` |
| New | `m11_fifa/quality.py`: the 5-class taxonomy and the acceleration flag (limit chosen from the empirical distribution and frozen in the protocol) |
| Outputs | `data/processed/m11-quality-flags-v1/<match>/flags.npz`, `summary.json`, `manifest.json` |
| Checks | Unit tests on synthetic tracks; flag counts reconcile with Step 1 (jumps 385,029; invalid steps 938); raw files unchanged (hash) |
| Runtime | About 1–2 h |
| Dependencies | Phase 1 |
| Pass / fail | Pass if counts reconcile exactly. Fail if any raw or v2 hash changes |
| Stop | Corruption class > 1 % of player-frames in any match (needs investigation first) |
| Approval | User (included in the CONDITIONAL GO) |

## Phase 3: Proximity generator, R ≤ 2 m

| | |
|---|---|
| Inputs | Phase 2 flags, raw cache |
| Code to reuse | `intersections.proximity` (`pair_rows`, `exact_border`, `qualify`, `episodes`, `context`), `ProximityConfig` with thresholds (0, 0.10, 0.25, 0.50, 1.0, 2.0) |
| New | A thin driver; band, tier and cluster columns; P(d ≤ R) descriptor |
| Outputs | `data/processed/m11-proximity-v4/<match>/episodes.jsonl`, `summary.json`, `manifest.json` |
| Checks | Episode counts at 0 / 0.10 / 0.25 / 0.50 m equal stored v2 **exactly** for all 40 matches (already verified for 3812 and 3816); symmetry test (swap a, b) |
| Runtime | About 40–60 min |
| Dependencies | Phase 2 |
| Pass / fail | Pass if v2 counts are reproduced exactly. Then report volumes per band, tier and cluster, and context completeness |
| Stop | Any v2 mismatch; < 50 % context completeness at 2 m |
| Approval | User. **Second approval gate before Phase 4** (volume and quality report) |

## Phase 4: Strata and descriptors (burst F, crossing Δt, crowding)

| | |
|---|---|
| Inputs | Phase 3 episodes |
| Code to reuse | Catalogue crossing logic (`intersections.catalogue`), `proximity_context` |
| New | Burst-F detector; crossing flag and \|Δt\| as metadata; crowding count within 2 m; |Λ| Kalman margin |
| Outputs | `m11-proximity-v4/<match>/descriptors.parquet` (or `.npz` if no parquet engine) |
| Checks | Burst-F rate reproduces 32–34 % within 1 m on 3 matches; Δt-vs-d_min table reproduces F3 on S10 |
| Runtime | About 1 h |
| Dependencies | Phase 3 |
| Pass / fail | Pass if the reproductions agree within ±2 percentage points |
| Stop | Reproductions disagree |
| Approval | User |

## Phase 5: Sample builder

| | |
|---|---|
| Inputs | Phases 3–4 |
| Code to reuse | `m11_recovery.views`, `m11_recovery.dataset` (only where its schema fits; otherwise new) |
| New | 15 Hz canonical resampling; event-centred normalisation; symmetric and independent gaps; seeded permutation target; metadata table |
| Outputs | `data/processed/m11-samples-v1/<match>.npz`, `meta.parquet`, `manifest.json` |
| Checks | Schema test: no ID, team or visibility in the inputs; keep/swap balance 50 ± 1 %; deterministic with the seed |
| Runtime | About 1–2 h |
| Dependencies | Phase 4 |
| Pass / fail | Pass if all schema and determinism tests pass |
| Stop | Any metadata leak into the inputs |
| Approval | User |

## Phase 6: Leakage and symmetry test suite

| | |
|---|---|
| Inputs | Phase 5 samples |
| Code to reuse | Pattern from `tests/test_m11_recovery.py` |
| New | `tests/test_m11_fifa_samples.py`: slot-order invariance; label-shuffle control; "distance only" shortcut probe; grouped-split integrity |
| Outputs | Tests only |
| Checks | Label-shuffle control AUC in [0.47, 0.53]; distance-only probe AUC ≈ 0.5 for the permutation target |
| Runtime | < 1 h |
| Dependencies | Phase 5 |
| Pass / fail | Pass if all tests pass |
| Stop | A shortcut probe gives AUC > 0.55 |
| Approval | — |

## Phase 7: Video-like noise and gap model

| | |
|---|---|
| Inputs | SoccerTrack **development** clips (117093, 118575, 118576); Wadi Degla tracker statistics (no labels) |
| Code to reuse | E1 bundles; `vision.pitch_mapping` |
| New | Empirical distributions of foot-point jitter, gap length and fragmentation; an augmentation module |
| Outputs | `data/processed/m11-noise-model-v1/params.json` |
| Checks | The noise parameters are fit on development clips only; augmented FIFA tracks match video jitter quantiles within 20 % |
| Runtime | About 2–3 h |
| Dependencies | Phase 5 |
| Pass / fail | Pass if the quantile match holds |
| Stop | Needs SoccerTrack validation data |
| Approval | User |

## Phase 8: M0 baselines

| | |
|---|---|
| Inputs | Phase 5 samples |
| Code to reuse | `m11_recovery.motion`, `m11_recovery.evaluate` |
| New | Nearest-end, constant-velocity and Kalman + Hungarian on (x, y, t) |
| Outputs | `data/processed/m11-fifa-baselines-v1/` |
| Checks | Nested grouped CV by match (40 matches, outer 5 folds); cluster-bootstrap CIs; metrics per band and tier |
| Runtime | About 1 h (CPU) |
| Dependencies | Phase 6 |
| Pass / fail | Reference only (no pass threshold) |
| Stop | — |
| Approval | **User (start of training = second gate)** |

## Phase 9: M2 sequence model (GRU / TCN)

| | |
|---|---|
| Inputs | Phases 5 and 7 |
| Code to reuse | `m11_recovery.models` training loop if compatible |
| New | Small GRU/TCN (≤ 1 M params) on (x, y, t, mask) with augmentation |
| Outputs | `data/processed/m11-fifa-m2-v1/` (OOF scores, configs, seeds) |
| Checks | Same CV as Phase 8; 3 seeds; per-band, per-tier and burst-F results |
| Runtime | About 3–6 h on GPU |
| Dependencies | Phase 8 |
| Pass / fail | Pass if accuracy beats Kalman by ≥ 2 points with the CI lower bound > 0 on T1+T2 (pre-registered in Phase 1) |
| Stop | No gain over Kalman → report and stop the ladder |
| Approval | User |

## Phase 10: M1 GBM on justified kinematics

| | |
|---|---|
| Inputs | Phase 5 |
| Code to reuse | `m11_recovery.experiments` |
| New | Feature list with a written justification for each feature; `HistGradientBoostingClassifier` |
| Outputs | `data/processed/m11-fifa-m1-v1/` |
| Checks | Same CV; feature schema asserted |
| Runtime | About 1 h |
| Dependencies | Phase 8 |
| Pass / fail | Comparative only |
| Stop | — |
| Approval | User |

## Phase 11: Ablations

| | |
|---|---|
| Inputs | Phases 8–10 |
| New | Training-subset ablations of equal size: ≤ 1 m vs ≤ 2 m; T1 vs T1+T2 vs all; with and without burst F; gap length; rotation; PCHIP sensitivity re-run of the generator |
| Outputs | `data/processed/m11-fifa-ablations-v1/` |
| Checks | Every ablation is evaluated on the same CEP |
| Runtime | About 6–10 h |
| Dependencies | Phases 9 and 10 |
| Pass / fail | Report only |
| Stop | — |
| Approval | User |

## Phase 12: Model freeze

| | |
|---|---|
| Inputs | Phases 8–11 |
| New | Frozen model card, weights hash, threshold and abstention rule |
| Outputs | `data/processed/m11-fifa-frozen-v1/` |
| Checks | Retrain on all 40 development matches with the frozen config; hash recorded |
| Runtime | About 1 h |
| Dependencies | Phase 11 |
| Pass / fail | Pass when frozen |
| Stop | — |
| Approval | **User + advisor** |

## Phase 13: FIFA validation (single disclosed use)

| | |
|---|---|
| Inputs | 12 validation matches (already used twice; disclosed) |
| Code to reuse | Phases 3–5 pipeline, frozen model |
| Outputs | `data/processed/m11-fifa-validation-v4/` |
| Checks | Run once; refuse to overwrite |
| Runtime | About 1 h |
| Dependencies | Phase 12 |
| Pass / fail | Pre-registered in Phase 1 |
| Stop | — |
| Approval | **Explicit user + advisor approval**. The test set is never reopened |

## Phase 14: Wadi Degla human labelling protocol (W2, fixed camera)

| | |
|---|---|
| Inputs | Wadi Degla fixed-camera videos; tracker output |
| Code to reuse | Multicam lab, review layer |
| New | Labelling protocol: about 60 random 10 s clips, double-labelled subset for agreement, and a labelling tool |
| Outputs | `data/processed/wadi-degla-identity-labels-v1/` |
| Checks | Inter-annotator agreement ≥ 0.8 on the double subset; clips are chosen randomly, not from model output |
| Runtime | About 25 person-hours **[I]** |
| Dependencies | Phase 1 (can run in parallel with Phases 8–12) |
| Pass / fail | Pass if agreement ≥ 0.8 |
| Stop | Agreement < 0.7 |
| Approval | **User** (this starts Wadi Degla processing) |

## Phase 15: Wadi Degla deployment evaluation and report

| | |
|---|---|
| Inputs | Frozen model, Wadi Degla tracker output resampled to 15 Hz, Phase 14 labels |
| Code to reuse | CV pipeline (ByteTrack B0), pitch mapping |
| New | Deployment adapter; evaluation against the labels; final report |
| Outputs | `data/processed/m11-wadi-degla-eval-v1/`, `docs/research/M11_FINAL_REPORT.md` |
| Checks | Compare against ByteTrack-only and Kalman baselines on the same labels; cluster CIs |
| Runtime | About 3–5 h |
| Dependencies | Phases 12 and 14 |
| Pass / fail | Pass if the frozen model reduces identity errors vs ByteTrack-only with a CI lower bound > 0 |
| Stop | No gain → report honestly as a negative transfer result |
| Approval | User + advisor |

---

## Approval gates (summary)

| Gate | Before phase | Approver |
|---|---|---|
| G1 | 1–3 (protocol, audit, generator) | User + advisor (pending now) |
| G2 | 4–8 (descriptors, samples, training start) | User, after the Phase 3 report |
| G3 | 12–13 (freeze, validation use) | User + advisor |
| G4 | 14–15 (Wadi Degla processing and labelling) | User |
