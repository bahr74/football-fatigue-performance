# M12 report 2: Scientific audit of the historical two-player result (v1)

Status: done. Date 2026-11-21. **Read-only on the historical outputs:** no metric, prediction or checkpoint was changed.

- Code: `python -m football_fatigue.m11_audit validity` (`src/football_fatigue/m11_audit/validity.py`, runtime about 10 s on CPU)
- Output: `data/processed/m11-scientific-audit-v1/{audit.json, manifest.json}`
- Tests: `tests/test_m11_audit.py` (8 tests)

**Scope.**
- **Data:** the 32 authorised train/CV matches, Stage 3 primary table `g0p80` (0.8 s gap, 2 s context, 15 Hz), Stage 4
  out-of-fold predictions.
- **Unit:** the 18,777 independent samples (one per candidate encounter).
- **Uncertainty:** every confidence interval is a match-level bootstrap (32 matches resampled, 1,000 draws).
- **Out of scope:** held-out matches and any official FIFA validation or test data.

**Question.** Is the historical GRU balanced accuracy (BA) of about 0.90 trustworthy, and what does it measure?

## Answer in one paragraph

The number is **real as a measurement**:
- no leak was found in splits, labels, early stopping, model selection, duplicate inputs or identity overlap;
- across seeds it is stable at 0.899 to 0.907.

It is **not a measure of identity recovery in real failures**:
- about two thirds of the samples have provider-ESTIMATED positions in the context;
- a label-free rule that only compares trajectory "texture" (how smooth the provider interpolation is) already reaches 0.68 BA
  overall and 0.72 when both players are ESTIMATED;
- where both players are VISIBLE inside the gap (10 % of samples), the GRU drops to 0.81 and its lead over HGB is not significant;
- when the texture is neutralised (smoothing or 5 cm noise at training time), the GRU is at or below HGB.

The GRU's advantage is therefore largely a better reading of provider interpolation artefacts, not better motion reasoning. That
is a property of this dataset, not of real video failures.

## A. Verified findings (data-checked; automated tests where marked)

| # | Check | Result | Test |
|---|---|---|---|
| A1 | Outer folds | 8 folds partition the 32 matches; every model's out-of-fold fold equals the frozen fold; every sample is predicted exactly once | yes |
| A2 | Early stopping | Neural early-stopping matches are never in the evaluated fold (all 8 neural variants checked) | yes |
| A3 | Tabular selection | Grid by inner CV on training folds only; SVM Platt scaling from inner out-of-fold | code read |
| A4 | Labels | y = `in_order XOR out_order` for every row; swapped share 0.4986 (per match 0.470 to 0.534); orders come from a hash of the candidate id, never from inputs | yes |
| A5 | Hidden interval | Changing in-gap vertices never changes model inputs; the frame is computed from incoming tracklets only; features and sequence tensors do not depend on y | yes |
| A6 | Structural leak probe | HGB on `n_valid_*` and `frame_mode`: BA 0.496 [0.492, 0.500]; logistic 0.499. No label signal in structure | yes |
| A7 | Duplicates | 0 exact duplicate inputs, 0 cross-match duplicates among 25,331 samples | no |
| A8 | Player overlap | 65 % of samples have both players seen in training, with no effect: GRU 0.9015 (both seen) vs 0.8981 (not both seen) | no |
| A9 | Seeds and selection | GRU seed BA 0.9003 / 0.9072 / 0.8987 (sd 0.0045); 14 learned variants compared; winner's-curse scale about 0.008 BA | no |
| A10 | Held-out guard | Loading a held-out match is refused before any data is read | yes |

### Overall (independent samples, n = 18,777)

| Model | BA [95 % CI] |
|---|---|
| GRU gap-last | **0.9003** [0.8954, 0.9048] |
| HGB | 0.8695 [0.8629, 0.8754] |
| SVM | 0.8586 [0.8525, 0.8645] |
| LSTM gap-last | 0.8583 [0.8536, 0.8629] |
| GRU − HGB (paired) | **+0.0308** [0.0252, 0.0364] |

### By provider status inside the hidden interval

| Stratum | Share | GRU | HGB | Forward Kalman | GRU − HGB [CI] |
|---|---|---|---|---|---|
| Both ESTIMATED | 52 % | 0.9115 | 0.8752 | 0.745 | +0.036 [0.028, 0.044] |
| Mixed | 38 % | 0.9083 | 0.8812 | n/a | +0.027 |
| **Both VISIBLE** | 10 % (n = 1,881) | **0.8118** | 0.7953 | n/a | +0.017 [−0.000, 0.031] |

### By ESTIMATED share of the context

| Stratum | Share | GRU | HGB | GRU − HGB [CI] |
|---|---|---|---|---|
| All visible | 4.9 % (n = 928) | 0.863 | 0.844 | +0.019 [−0.005, 0.044] |
| ≤ 50 % ESTIMATED | 28 % | 0.879 | n/a | +0.019 |
| > 50 % ESTIMATED | 67 % | 0.912 | n/a | +0.037 |

### Other strata

| Stratum | n | GRU | HGB | Forward Kalman | GRU − HGB [CI] |
|---|---|---|---|---|---|
| Local crossing | 12,193 | 0.914 | 0.889 | 0.645 | +0.025 [0.018, 0.032] |
| No crossing | 3,111 | 0.858 | 0.809 | 0.697 | +0.049 [0.034, 0.064] |
| Clone-like pair | 1,032 | 0.895 | 0.838 | 0.707 | +0.056 [0.027, 0.082] |
| Label-risk | 269 | 0.856 | 0.841 | 0.540 | +0.015 [−0.036, 0.061] |
| Closest distance < 0.05 m | 2,129 | 0.889 | 0.850 | 0.603 | +0.040 [0.023, 0.054] |

### Shortcut probes (label-free rules and texture-only models, grouped by outer fold)

| Probe | All | Both ESTIMATED | Both VISIBLE |
|---|---|---|---|
| Anti-nearest rule (pick the assignment whose endpoints are farther apart) | 0.646 [0.638, 0.655] | n/a | n/a |
| Texture similarity rule (no labels, no training) | **0.680** | **0.719** | 0.540 |
| Texture-only HGB probe | 0.679 | 0.715 | 0.539 |

The median per-step texture (second difference) is 0.070 m for ESTIMATED and 0.017 m for VISIBLE samples: the provider fills
ESTIMATED stretches with visibly different motion.

### Existing robustness outputs (Stage 4, re-read, not recomputed)

| Condition | GRU | HGB |
|---|---|---|
| Inference on smoothed input (smooth5) | 0.836 | 0.864 |
| Inference with 5 cm noise | 0.786 | 0.851 |
| Retrained on smooth5 | 0.853 | 0.869 |
| Retrained on 5 cm noise | 0.850 | 0.853 |
| Retrained on smooth5, both VISIBLE | 0.767 | 0.784 |
| Centred-trained, shifted gap | 0.822 | 0.818 |
| Dependency weighting (change) | −0.016 | −0.003 |

Stage 4C shift-mix GRU: 0.926 centred, 0.910 shifted. Same texture caveat.

## B. Suspected risks (plausible, not proven)

1. **Selection on provider chain continuity.** 3,892 Stage 2 candidates were rejected because of a broken chain. The retained
   events are the ones where the provider already kept identity, so they may be easier than events where the tracker itself
   failed.
2. **Centring prior.** The gap is centred on the closest approach, so models can learn where in the window the crossing happens.
   Shifted-gap evaluation costs 0.08 BA for the centred-trained GRU.
3. **ESTIMATED interpolation as shortcut.** It explains a large part, but not all, of the GRU's lead: the lead stays positive,
   though not significant, in both-VISIBLE.
4. **Dependence.** Samples sharing a player within seconds are correlated; the match-level bootstrap covers between-match
   variation, not within-match clustering. Dependency weighting moved the GRU by −0.016.
5. **Proximity is not occlusion.** No event is verified as a visual occlusion; real video failures will have detector and
   calibration noise (decimetres to metres), which is far larger than the provider noise these models saw.

## C. Unresolved questions

- How much of the remaining both-VISIBLE lead (+0.017) is real motion skill versus residual texture?
- What BA does a model reach on human-verified real tracker failures? There is none yet: no human-reviewed set exists.
- Does the GRU lead survive a training distribution that randomises texture (`robust_mix`), in a fair compute-matched search?
  This is the GPU pilot question.

## D. Recommended experiments (in order)

1. Report every future two-player result **stratified by visibility**: both VISIBLE, and contexts that are all visible. Make
   both-VISIBLE the headline, not the pooled BA.
2. **GPU pilot** (report 4): GRU vs parameter-matched LSTM on `centred` and `shift_mix`, one fold, to measure convergence and cost.
   Then a compute-matched search with the `robust_mix` distribution (noise and dropout), so texture cannot be read.
3. Add the label-free texture rule as a **mandatory reference baseline** in every table.
4. Human review of the both-VISIBLE and clone-like strata before any v2 freeze (report 8).
5. Evaluate on shifted, variable-gap and noisy stress tables, never only on centred `g0p80`.

## Automated tests added (`tests/test_m11_audit.py`)

- In-gap vertices never change inputs.
- Label equals the order bits, and the frame ignores outgoing tracks.
- Features and sequence tensors do not depend on y.
- Roughness (texture) definition.
- Match-level bootstrap.
- Saved folds and labels are valid (data).
- Structural inputs carry no label signal, and the texture rule is stronger on ESTIMATED than VISIBLE (data).
- Held-out access is refused.
