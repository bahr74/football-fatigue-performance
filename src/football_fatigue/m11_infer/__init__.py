"""M11 inference: loading the Stage 4C GRU C / HGB C models, a timestamped-trajectory adapter and a routed, abstaining
two-in / two-out prediction API (Milestone 1, model readiness).

Modules:

- ``bundle``: per-fold checkpoints of GRU C (seed 0, 160-epoch budget) and HGB C with metadata, hashes and reproduction checks.
- ``adapter``: timestamped pitch-coordinate tracklets -> the unchanged Stage 3 model inputs (same grid, masks and frame).
- ``predict``: routing policy ``m11-routing-v1``, abstention, the ``Prediction`` result and the ``Predictor`` API.
- ``fifa``: development-only source of real FIFA trajectories (train/CV matches; held-out matches are refused).
- ``__main__``: ``python -m football_fatigue.m11_infer {check-bundles, check-adapter, export-fifa, predict}``.

Probabilities are always p = P(SWAPPED) (Stage 3 convention). No model here is trained on all 32 matches: every bundle is one
outer-fold model and is validated only by its out-of-fold predictions.
"""
