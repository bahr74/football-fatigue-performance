"""Fair GRU / LSTM / SVM / HGB benchmark infrastructure (configurable architectures, seeded equal-budget searches, match-disjoint
fit / early-stop / calibration / test roles, convergence diagnostics, calibration, paired match-level confidence intervals).

Historical code and checkpoints are untouched: ``m11_benchmark.models.make_net`` keeps its fixed sizes; this package builds its own
configurable networks (identical to the historical ones at the historical settings, which is tested).

GPU training is gated: ``python -m football_fatigue.m11_fair train`` refuses to run without an explicit approval id."""
