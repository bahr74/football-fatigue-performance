# M11 identity preservation and recovery — progress log (autonomous run)

Living document: completed stages, findings, open questions, experiment results, next actions and reproduction commands.
Development matches only for iterative work: 3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506 (the ten with Step 1
caches). Official validation (12 matches) is reserved for one final evaluation of a frozen model; the locked test (12) is never
accessed. Nothing is committed.

## Stage log

| stage | status | outputs | key result |
|---|---|---|---|
| v1 geometric detector (3812) | done earlier | `m11-intersections-v1/3812` | 18,469 path intersections |
| S1 temporal audit v2 (3812) | done earlier | `m11-intersections-v2/3812` | 180 clean @0.20 s; spike gap found |
| **S1.1** audit v3, 10 dev matches | done | `m11-intersections-v3/<match>`, `.../cross-match` | 3812 events reproduce v1 byte-for-byte; all v2 differences explained by the spike rule (clean @0.20: 180 → 164) |
| S1.1 totals | done | `m11-intersections-v3/cross-match/summary.json` | 166,819 intersections; clean@0.20 1,382 (per match 107–180); invalid ≤ 2.5 % (mostly duplicate-position artefacts, e.g. 3814: 254 of 300) |
| **S1.2** identity-free contacts | done | `m11-contacts-v1/<match>` (10 matches) | 24,685 contacts; @0.20 clean 5,039, uncertain 18,648, invalid 998; candidate rows by geometry 46 % no crossing, 36 % crossing not aligned, 18 % advisor aligned crossing |
| E1 baselines + GBM (LOMO) | done | `m11-recovery-v1/experiments/e1-baselines-gbm` | baselines 0.48–0.56; GBM 0.764, FA5 % cov 0.378 |
| E2 hybrid (LOMO, GPU) | done | `experiments/e2-hybrid` | 0.769, log loss 0.453 (not significantly better than GBM) |
| R1 gap-centre robustness | done | `experiments/r1-shifted` | GBM 0.761 with shifted gaps (no t* leak) |
| Selection + freeze | done | `models/gbm-v1` (selected); comparators `hybrid-v1`, `ca_bidir-v1`, `nearest-v1` | GBM by the pre-registered rule |
| Official validation (once) | done | `validation/final`, `validation/addendum-nearest-baseline` | GBM 0.767; frozen FA5 %: FA 0.048, cov 0.377, rec 0.359 |
| API, dashboard, figures, report | done | `m11_recovery/{infer,__main__,app,figures}.py`, [report](m11_identity_recovery_report.md) | API = batch prediction exactly |

**Checkpoint.** All stages are complete and nothing is committed. Next steps need the advisor (see open questions): a real
tracker-gap test, human annotation at contacts, and the N-player extension.

## Findings

1. **Speed spikes** are position noise, not exchanges (3812: 32 % within ±0.5 s of the crossing, median separation 3.5 m,
   4/185 land towards the other player). Rule: a spike within ±0.5 s of a player's crossing makes the event uncertain.
2. **Selection by path intersection leaks the label.** Whether two paths cross is computed from identity-labelled
   trajectories. Among time-aligned intersections the nearest-position rule agrees with the label only 22–29 % of the time
   (3812, all gaps), i.e. "assume they crossed" gives ~75 % with no motion model. At inference a tracker does not know whether
   meeting players crossed or turned back. **Correction (S1.2):** samples are defined identity-free by *contact* (same-time
   distance <= 1.0 m, episode while <= 1.5 m, time of minimum distance); geometric crossing is a stratum. On 3812 51 % of
   contact candidates cross and 49 % do not; the advisor's time-aligned crossings are a reported subset (500 contacts).
3. **Provider jitter at contact.** At contact the RAW positions jitter by up to ~0.7 m between 10 Hz grid points; short-span
   motion fits are unreliable there (constant velocity over 0.5 s: 0.57–0.65 agreement on the biased set at zero noise).

## Pre-registered protocol (written before any learned-model result)

- **Population.** All contact candidates of the 10 development matches at τ = 0.20 s that are clean or uncertain (invalid
  tracking artefacts are excluded by the predefined S1 rule, counted). Tier (clean/flagged) and geometry (no crossing /
  crossing not aligned / advisor aligned crossing) are strata, never filters.
- **Scenarios.** 5 gaps (0.4–2.0 s) × 3 noise levels (0, 0.15, 0.3 m) per interaction; samples with < 3 observed points per
  track are counted as unusable per scenario.
- **Validation inside development.** Leave-one-match-out (10 folds); no row of the held-out match enters fitting, early stopping
  or scaling. Pooled out-of-fold predictions are scored, plus a per-match breakdown and 95 % match-bootstrap intervals.
- **Model selection rule.** Primary metric: pooled out-of-fold log loss. The selected model is the simplest model
  (baseline < GBM < hybrid) whose paired match-bootstrap 95 % interval of log-loss difference against the best model
  includes 0; otherwise the best model.
- **Operating thresholds.** The selected model's confidence thresholds for 1/2/5/10 % false association are fitted once on its
  pooled development out-of-fold predictions and frozen; default operating point 5 %. The final model is refitted on all 10
  development matches with fixed hyperparameters.
- **Official validation.** One evaluation of the frozen model and thresholds; no change afterwards. This partition was examined
  earlier (Step 6 of the historical pipeline), so it is not pristine. The locked test partition is not used.
- **Leak oracle.** "Assume they crossed" is reported on the advisor subset only. It uses the identity-derived crossing flag and is
  not a valid predictor.

## Open questions for Dr. Basha

- Acceptance of identity-free contact selection (r = 1.0 m) instead of geometric intersection for model samples, keeping the
  intersection catalogue for analysis.
- Provisional τ = 0.20 s for the advisor subset; provider continuity as reference; ESTIMATED-heavy contacts.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.audit_v3 --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
.\.venv\Scripts\python.exe -m football_fatigue.intersections.contacts --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
```

## Advisor correction (2026-11-09) — exact geometric intersection

See [m11_advisor_intersection_correction.md](m11_advisor_intersection_correction.md).

- The v1/v3 detector **is** the advisor's algorithm. Verified against an independent implementation of his
  project/searchsorted recipe on all 166,819 development intersections: points within 7e-15 m; times identical except 7
  first-nearest-pass ambiguities of `project`.
- Players do not generally reach P together: median |t_A − t_B| 2.16 s; 4.2 % within 0.20 s.
- Advisor population (aligned intersections), leave-one-match-out:
  - "assume crossed" 0.750;
  - GBM trained there 0.797;
  - contact-trained GBM 0.707.
- The advisor-trained GBM is at chance (0.507) on non-crossing contacts, so it learned the selection prior.
- New experiments: `a1-advisor-intersections`, `a2-transfer-contacts-to-intersections`, `a3-transfer-intersections-to-contacts`.
  Contact-formulation results are kept and labelled as such.
- **Next (needs advisor approval):** a label-symmetric advisor population (aligned exact intersections + matched non-crossing
  meetings), then retraining on development. Optional: a Shapely install for a literal cross-check. No validation or test use
  without approval.

## Re-alignment with the advisor's extraction procedure (2026-11-09)

See [m11_occlusion_catalogue_realignment.md](m11_occlusion_catalogue_realignment.md). Scope: occlusion dataset preparation only
(no training).

- New output `m11-occlusion-catalogue-v1`: 166,819 events on 10 development matches.
  - Byte-identical to v3, with full window provenance, Before/After reconstruction and threshold-free timing evidence.
  - The 0.20 s `simultaneous` flag is dropped.
- Counts: 231 pairs per window (not 121); windows are 5 s.
- Same time at P:
  - 15 exact ties;
  - 2.1 % with concurrent crossing segments;
  - median |Δt| 2.16 s.
- Open: the advisor's same-time criterion (question 1 of that report).

## Spatiotemporal occlusion-proxy dataset (2026-11-09)

See [m11_spatiotemporal_occlusion_dataset.md](m11_spatiotemporal_occlusion_dataset.md). Project decision: a crossing counts only
with temporal coincidence.

- Rule: rejected if |Δt| > max(h, 2σ_dt); accepted if |Δt| ≤ 2σ_dt ≤ h, well conditioned, with no missing sample and no veto;
  otherwise ambiguous. h = 0.0667 s, the native update interval.
- Output `m11-occlusion-spatiotemporal-v1` (10 dev matches):

  | accepted | ambiguous | rejected |
  |---|---|---|
  | 284 (103 with both observed) | 8,841 | 157,694 |

- Sensitivity: 0 to 1,454 accepted across the grid.
- Smoothing study (SG5): removes 21.7 % of candidates and 0 accepted. Smoothing is not applied.
- Next: advisor confirmation of τ / z, of ESTIMATED crossings, and of a label-symmetric training counterpart.

## Exact spatiotemporal extraction (2026-11-10)

See [m11_exact_spatiotemporal_extraction.md](m11_exact_spatiotemporal_extraction.md). The rule is A(t*) = B(t*), solved exactly
(integer µs/mm plus `Fraction`), with no tolerance, distance, speed, angle, smoothing or confidence threshold.

- Output `m11-exact-spatiotemporal-v1` (10 development matches).
- Counts:
  - 14,321,863 pair-window comparisons;
  - 166,819 2D crossings;
  - 139,471 exact pair-interval solutions;
  - 12,004 window-level detections;
  - **566 unique events**: 15 interior points and 551 coincident intervals.
- The 551 intervals are same-team ESTIMATED duplicate coordinates, a provider artifact.
- The 15 points equal the catalogue's 15 ties and follow the equal-x lattice pattern.
- A ±0.5 mm perturbation leaves 0 solutions.
- Conclusion: simultaneous same-point occupancy is non-generic, so the advisor's code (2D) and his "same time" sentence cannot
  both hold without a tolerance.
- **Next (needs the advisor):** choose 2D crossings, an advisor-stated tolerance, or same-time proximity.
- **Needs authorization:** Step 1 caches for the 30 remaining development matches; validation and test need explicit
  authorization.

## Synchronized proximity encounters (2026-11-10)

See [m11_sync_proximity_encounters.md](m11_sync_proximity_encounters.md). Criterion: d(t) <= R at the same t. The primary R is
0.25 m, with sensitivity at 0, 0.10 and 0.50 m. The minimum is solved analytically per pair-interval (with an exact border
re-decision), and episodes are merged per pair.

- Output `m11-sync-proximity-v1` (10 development matches).
  - Unique episodes: 566 / 3,596 / **8,215** / 16,583 at 0 / 0.10 / 0.25 / 0.50 m.
  - 3812: 18 / 327 / 844 / 1,737.
- Reconciliation:
  - R = 0 equals the exact solver;
  - the historical 13,786 is reproduced exactly; it counts synchronous-interval runs only;
  - all 18 exact events of 3812 lie inside 0.25 m episodes.
- The 551 coincident intervals are cloned provider estimates: all 22 players are ESTIMATED and one estimated track is copied
  onto a teammate.
- 74 % of 0.25 m episodes are estimated/estimated (baseline 42–58 %) and 829 are observed/observed.
- Position noise:
  - VISIBLE p90 is 2–3 cm, so 0.10 m is resolvable;
  - ESTIMATED p90 is 10–15 cm.
- Recommendation: keep 0.25 m, stratify by `pair_quality` and exclude coincident intervals from training.
- Next (needs approval): build Step 1 caches for the 30 other development matches; the training design. Validation and test need
  authorization.

## Complete development encounter dataset (2026-11-10)

See [m11_sync_proximity_dev_dataset.md](m11_sync_proximity_dev_dataset.md).

- `m11-sync-proximity-v2` covers all 40 development matches: 30 Step 1 caches were built and all 40 matches were extracted, with
  0 failures.
- Episodes are identical to v1. v2 adds context completeness, cut reasons and deterministic hashed outputs.
- 0.25 m results:
  - 30,438 episodes;
  - by quality: 2,619 observed/observed, 4,761 mixed, 23,058 estimated/estimated;
  - 2,697 coincident intervals (provider cloning);
  - 1,917 high-quality candidates;
  - context complete 24,088 / incomplete 6,350.
- Sensitivity: 2,750 / 13,915 / 30,438 / 61,106 episodes at 0 / 0.10 / 0.25 / 0.50 m.
- Index, subsets, validation (29 checks per match, all pass), determinism (3 matches byte-identical) and schema are in
  `m11-sync-proximity-v2/development-index/`.
- Validation and test were not touched, and there is no training.
