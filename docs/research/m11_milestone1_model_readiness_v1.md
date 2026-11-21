# M11 Milestone 1: model readiness (`m11_infer`)

**Status:** implemented and tested.
- No training was run, and no model, dataset, split or earlier output was changed.
- No held-out or official FIFA validation/test data were read.
- **Data used:** the 32 train/CV development matches only.
- **Model status:** no model trained on all 32 matches exists yet. Every bundle is one outer-fold model, validated only by its out-of-fold predictions.

## 1. What was built

**Package `src/football_fatigue/m11_infer/`:**

| Module | Purpose |
|---|---|
| `bundle.py` | Loads the existing Stage 4C fold checkpoints: GRU C (`neural/C/seed0/fold<f>_c160.pt`) and HGB C (`hgb/C/fold<f>.joblib`). Handles fold/match bookkeeping, refuses held-out matches, checks SHA-256 hashes, and reproduces stored out-of-fold predictions. Also recomputes the abstention thresholds. |
| `adapter.py` | Converts timestamped pitch-coordinate tracklets into the unchanged Stage 3 model arrays (grid, masks, frame). |
| `predict.py` | Routing policy `m11-routing-v1`, abstention, the `Prediction` result and the `Predictor` API. |
| `fifa.py` | Development-only source of real FIFA scenarios, plus the adapter-from-raw reproduction check. |
| `__main__.py` | Command line: `check-bundles`, `check-adapter`, `export-fifa`, `predict`. |

**Recorded once:** checkpoint hashes and the full check report, in `data/processed/m11-infer-v1/bundles_v1.json`.

## 2. Reproduction results

Command: `python -m football_fatigue.m11_infer check-bundles --device cpu --write`. It ran on all 25,331 centred 0.8 s samples (each fold model on its own unseen matches).

| Model | Max \|Δp\| vs stored OOF | Decision flips | Abstention flips (5 %) | Outgoing-swap symmetry error | Fold BA range |
|---|---|---|---|---|---|
| GRU C (seed 0, 160-epoch budget) | 9.1e-5 (CPU) | 0 | 0 | ≤ 4.8e-7 | 0.915–0.938 |
| HGB C | 0.0 | 0 | 0 | ≤ 1.1e-16 | 0.854–0.868 |

**Tolerances**, derived from each model's implementation and measured:
- **GRU:** float32, and the stored values were computed on CUDA. On CUDA with the same batches the reproduction is exact (0.0). One sample at a time on CUDA differs by 3.0e-7, so the CUDA tolerance is 1e-6. On CPU the summation order differs, reaching 9.1e-5, so the CPU tolerance is 2.5e-4.
- **HGB:** float64 and deterministic, so the tolerance is 1e-12.
- **Both:** additionally, no decision may flip.

**Thresholds recomputed** from the saved out-of-fold predictions (independent subset, all folds) match the documented values:

| Model | 5 % error threshold | Coverage | 2 % error threshold | Coverage |
|---|---|---|---|---|
| GRU C | 0.6425 | 0.927 | 0.8525 | 0.788 |
| HGB C | 0.7875 | 0.716 | 0.9029 | 0.546 |

**Adapter check:** rebuilding Stage 3 samples of match 3812 from raw provider trajectories through the adapter (`check-adapter --match 3812 --limit 60`) gave 60/60 byte-identical model arrays. Predictions equal the stored out-of-fold values within tolerance: GRU 1.8e-5 on CPU and 3.0e-7 on CUDA; HGB 0.0.

**GRU C budget:** the epoch budget was binding in folds 0, 3 and 4 of the seed-0 bundles; see [the correction note](m11_stage4c_correction_v1.md).

## 3. Input contract and supported conditions

**Input:**
- Two incoming and two outgoing tracklets, each with times in seconds (strictly increasing) and x, y in metres in one metric pitch frame.
- The hidden interval `[g0, g1]`, or `gap: {"from_observations": true}`.
- Optionally `noise_sigma_m`, the declared positional noise.

**Sampling:** by timestamp onto the 15 Hz, 2.0 s Stage 3 grid. There is no extrapolation, and observations inside the gap are ignored and counted. Exact gap semantics are in the correction note §2.

**Output:**
- `decision`: MATCHED, SWAPPED or ABSTAIN; `predicted_pairing`; `pairs`.
- `p_swapped`, `confidence` (max(p, 1 − p)) and `threshold`.
- `support`: primary, fallback or unsupported; `model`; `reasons`.
- Both models' probabilities; the fold and checkpoint hashes; an input summary.

**Routing policy `m11-routing-v1`** (declared, conservative, to be revised with Milestone 2 evidence):

| Condition | Support | Model | Evidence |
|---|---|---|---|
| Every tracklet has ≥ 3 grid points, gap 0.75–0.85 s, noise ≤ 2 cm or not declared | primary | GRU C | Out-of-fold at exactly 0.8 s on clean FIFA data: centred BA 0.924–0.929, shifted 0.909–0.911. The ±0.05 s band is a declared tolerance, not measured. |
| Gap in [0.4, 1.6] s outside that band, or noise in (2, 5] cm | fallback | HGB C | Measured transfer at 0.4 s (0.928) and 1.6 s (0.803) and at 5 cm noise (0.842) only. The threshold was calibrated at 0.8 s, so realised error elsewhere is unknown. |
| Gap < 0.4 s or > 1.6 s; noise > 5 cm; fewer than 3 valid grid points | unsupported | — | Never evaluated. The decision is ABSTAIN; probabilities are reported as diagnostic only. |
| Confidence below the chosen model's threshold | (any) | — | ABSTAIN, with reason `low_confidence`. |

**Not supported:**
- More or fewer than two players.
- One player hidden while the other stays visible.
- A missing outgoing tracklet, or an anonymous tracklet that belongs to neither player.
- Team or appearance cues.
- Video positional noise (calibration error about 0.4–1 m), which is far above the 5 cm evaluated.

## 4. Choosing a model for prediction

- `Predictor.for_match(m)` uses the fold models that never saw development match `m`. Results on that match are therefore unseen-match out-of-fold results, not independent validation.
- `Predictor(fold)` uses a chosen fold for external input. The fold models differ only in their training matches.
- A model trained on all 32 matches is Milestone 4. It needs approval and will still have no unbiased score until the approved final evaluation.

## 5. Tests

`tests/test_m11_infer.py` has 38 tests:
- **Unit:** the adapter reproduces `build_one`; observations inside the gap are ignored; no extrapolation; resampling at 25, 30, 50 and 60 fps; malformed input is rejected; gap and JSON handling; routing table; abstention at both operating points; fallback; unsupported input; too few points.
- **Integration (development data):** held-out and official matches are refused (by the API and by the FIFA source); checkpoints load and reproduce stored predictions; symmetry; a hash mismatch is refused; adapter rebuild from raw; symmetry on a real scenario; command-line prediction.

## 6. Commands

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_infer check-bundles --device cpu
.\.venv\Scripts\python.exe -m football_fatigue.m11_infer check-adapter --match 3812 --limit 60
.\.venv\Scripts\python.exe -m football_fatigue.m11_infer export-fifa --match 3812 --row 0 --out input.json --truth truth.json
.\.venv\Scripts\python.exe -m football_fatigue.m11_infer predict --json input.json            # fold from source.match_id
.\.venv\Scripts\python.exe -m football_fatigue.m11_infer predict --json input.json --noise-sigma-m 0.03
```
