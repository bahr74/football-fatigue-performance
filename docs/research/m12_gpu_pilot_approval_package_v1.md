# M12 report 4: GPU pilot approval package (v1)

Status: **PROPOSED, awaiting explicit approval. Nothing has been trained on the GPU.** Date 2026-11-21.

- Plan file: `data/processed/m11-fair-gpu-pilot-plan-v1/pilot_plan.json`, written by
  `scripts\session_a_python.cmd -m football_fatigue.m11_fair pilot-plan` (CPU, seconds).
- Code: `src/football_fatigue/m11_fair/gpu_pilot.py`.

## Purpose

Before asking for a full fair-benchmark budget, measure on the real GPU:

- s/epoch;
- peak GPU memory;
- best epoch under the cap;
- overfitting and at-cap behaviour.

The pilot covers the historical sizes, a parameter-matched LSTM and the largest search sizes. It produces **no model ranking**;
every output is labelled PILOT.

## Exact configurations

All configurations share these settings:

- Adam, lr 1e-3, weight decay 1e-4, batch 256, dropout 0.1;
- embedding 48, head 96;
- epoch cap 400, patience 25;
- early stopping on the 3 `early_stop` matches of the fold;
- temperature fitted on the `calibration` matches.

| config_id | Family | Hidden | Layers | Parameters | Training distribution | Why | Est. s/epoch |
|---|---|---|---|---|---|---|---|
| pilot-gru-h48-l1-centred | GRU | 48 | 1 | 43,585 | centred | Historical GRU | 1.24 |
| pilot-lstm-h40-l1-centred | LSTM | 40 | 1 | 41,985 | centred | Historical LSTM | 3.15 |
| pilot-lstm-h42-l1-centred | LSTM | 42 | 1 | 43,585 | centred | Parameter-matched to GRU-48 | 3.21 |
| pilot-gru-h96-l2-centred | GRU | 96 | 2 | 203,137 | centred | Largest GRU in the search | 2.68 |
| pilot-lstm-h96-l2-centred | LSTM | 96 | 2 | 259,969 | centred | Largest LSTM in the search | 7.84 |
| pilot-gru-h48-l1-shift_mix | GRU | 48 | 1 | 43,585 | shift_mix | Distribution cost | 1.24 |
| pilot-lstm-h42-l1-shift_mix | LSTM | 42 | 1 | 43,585 | shift_mix | Distribution cost | 3.21 |

- **Folds:** 1 (outer fold 0 of the frozen 8; train/CV matches only).
- **Seeds:** 1 (seed 0).
- **Search:** none, a single configuration at the full cap (`search="full"`).

## Estimates [EST], to be replaced by the pilot's measurements

**Basis.**
- Historical measured GPU s/epoch at the historical sizes (GTX 1650 Ti; GRU 1.24, LSTM 3.15).
- Scaled by √(parameter ratio) for the larger networks. This scaling is an assumption.
- **CPU timing is not used:** on CPU the GRU is slower than the LSTM, the opposite of the GPU.

| Quantity | Estimate |
|---|---|
| GPU hours if every configuration runs to the 400 cap | **2.5 h** |
| GPU hours if best epochs are about 225 (stop about 250) | **1.6 h** |
| Peak GPU memory | < 1.5 GB (inputs 46 MB centred / 369 MB shift-mix; networks < 0.3 M parameters) |
| Host RAM | about 1.3 GB peak (measured on CPU with the same pools) |

## Stop conditions (enforced by the runner)

1. **GPU busy:** another compute process is on the GPU (for example Session B) → **refuse to start**.
2. **Non-finite loss:** the run stops with an error.
3. **Too slow:** after a 5-epoch probe, s/epoch > 3× the estimate → that configuration is skipped and recorded.
4. **Memory:** peak GPU memory > 3,500 MB → the pilot stops.
5. **Wall clock:** more than 4 h → the remaining configurations are not started (recorded).

## Resume strategy

Re-run the same command. Per-epoch training states (`fold0_search/<config>.state.pt`) are reused, finished folds (`fold0.json`)
are skipped, and checkpoints are write-once with sha256 sidecars. Resumed training is bit-identical (tested on CPU).

## Outputs (`data/processed/m11-fair-gpu-pilot-v1/`)

- `plan.json` (with the approval id);
- `ledger.json` (per configuration: status, measured vs estimated s/epoch, epochs run, best epoch, at-cap, overfitting, peak GPU
  memory, GPU seconds);
- per configuration, `fold0.json` and `fold0_seed0.json` (history, convergence diagnostics, test predictions on all 6 stress
  tables);
- `checkpoints/*.pt` plus sidecars.

## Command (only after approval)

```bat
scripts\session_a_python.cmd -m football_fatigue.m11_fair gpu-pilot --approval <approval-id> --device cuda
```

Without `--approval` the command prints REFUSED and exits with status 2 (verified).

## Decisions the pilot enables

| Observation | Decision |
|---|---|
| Best epochs well below 400 for all configurations | Keep cap 400 and halving rungs (20, 60, 180, 400) |
| Any configuration at the cap and still improving | Raise the cap for that family, or report it as budget-limited |
| Measured LSTM/GRU s/epoch ratio | Sets the compute-matched configuration counts (`configs_for_budget`) |
| shift_mix cost vs centred | Sizes the `robust_mix` budget |
