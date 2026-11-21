# M11 Stage 4C: correction note v1 (GRU epoch budget and exact gap semantics)

This note corrects and completes [m11_stage4c_robustness_and_model_selection_v1.md](m11_stage4c_robustness_and_model_selection_v1.md). That report is kept unchanged. No result is recomputed or changed here: every number comes from the saved fold records in `data/processed/m11-stage4c-robustness-v1/neural/<config>/seed<s>/fold<f>.json` (`caps.160.epochs_run`, `caps.160.best_epoch`).

## 1. The 160-epoch budget is binding for GRU C

The Stage 4C report §3 lists config C at the 160-epoch cap as "5 / 8 folds stopped by patience (seed 0)". The summary text also says early stopping "now ends most fold fits before the cap". **Both statements hold for config A and for seed 0 of config C. They do not hold for config C across seeds.** In the table, a run is *binding* when it used all 160 epochs, i.e. patience did not stop it first.

| Config, seed | Folds that used all 160 epochs | Best epoch per fold (0–7) |
|---|---|---|
| A, seed 0 | 0 / 8 | 126, 118, 97, 113, 127, 101, 103, 92 |
| A, seed 1 | 0 / 8 | 86, 93, 104, 111, 83, 107, 93, 93 |
| A, seed 2 | 1 / 8 | 105, 159, 117, 127, 116, 103, 100, 78 |
| **C, seed 0** | **3 / 8** (folds 0, 3, 4) | 159, 114, 123, 157, 159, 143, 136, 130 |
| **C, seed 1** | **3 / 8** | 160, 157, 148, 156, 141, 148, 135, 123 |
| **C, seed 2** | **7 / 8** | 154, 155, 128, 158, 157, 160, 153, 157 |

**Consequences:**
- GRU C, the proposed primary candidate, is probably **under-converged**. Its validation loss was still improving at the cap in 13 of 24 fold fits. The development results are therefore a lower bound for this architecture and distribution, not a converged estimate.
- The 160-epoch budget and config C were both chosen after inspecting the same out-of-fold results that are reported. **The GRU C figures (centred BA 0.926, shifted 0.910) are therefore selection-optimistic.** This is distinct from, and in addition to, the binding cap.
- The Milestone 1 bundles use seed 0 (`fold<f>_c160.pt`). Folds 0, 3 and 4 of these are budget-bound. This is recorded per bundle (`budget_binding`) in `data/processed/m11-infer-v1/bundles_v1.json`.
- Any final training (proposed Milestone 4) needs a larger budget, chosen on the 32 development matches with nested selection. No retraining has been done.

## 2. Exact gap semantics

Code: `src/football_fatigue/window_intersections/samples.py`, constants `IN_REL` / `OUT_REL`, functions `tracklet` and `build_one`.

- **Hidden interval:** `[g0, g1]` with `g0 = tc - g/2` and `g1 = tc + g/2`. The model input `gap_s` equals `g` (0.8 for the primary table).
- **Incoming grid:** `g0 + k/15` for k = -30 … -1, so the last incoming grid point is `g0 - 1/15`.
- **Outgoing grid:** `g1 + k/15` for k = 1 … 30, so the first outgoing grid point is `g1 + 1/15`.
- **Observations used:** incoming positions interpolate only provider observations strictly before `g0`; outgoing positions only observations strictly after `g1`.
- **Unobserved duration on FIFA data:** provider observations arrive every 1/15 s. So for `gap_s = 0.8` the time between the last incoming observation and the first outgoing observation is between 0.80 and 0.93 s, depending on where the observation instants fall relative to `g0` and `g1`. The interval between the last incoming and first outgoing *grid point* is exactly 0.8 + 2/15 ≈ 0.933 s.
- **Gap length is never learned:** all training samples have `gap_s = 0.8`, including the shift-augmented ones. The sequence `dt` channel is the same fixed grid for every sample. The GRU has therefore never seen any other value of its gap input: results at 0.4 s and 1.6 s are extrapolation.
- **For real (video) tracks:** `m11_infer.adapter.gap_from_observations` sets `g0` and `g1` just inside the last incoming and first outgoing observation times. `gap_s` is then the unobserved duration. Under the routing policy, GRU C is used only for `gap_s` within 0.75–0.85 s; this band is a declared tolerance, not a measured one.
