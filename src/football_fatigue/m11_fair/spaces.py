"""Seeded, equal-size hyperparameter searches per family (scrambled Sobol in the unit cube, mapped to each space).

Every family receives the same number of configurations (``N_CONFIGS``). A selected value on the edge of a continuous range, or the
largest / smallest value of a discrete list, is flagged (``edge_flags``) and triggers one documented extension round."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

N_CONFIGS = 24

# name -> list of (param, kind, spec). kind: "log" (lo, hi), "lin" (lo, hi), "int_log" (lo, hi), "choice" [values]
SPACES: dict[str, list[tuple[str, str, Any]]] = {
    "gru": [("lr", "log", (3e-4, 3e-3)), ("hidden", "choice", [32, 48, 64, 96]), ("layers", "choice", [1, 2]),
            ("dropout", "lin", (0.0, 0.3)), ("weight_decay", "log", (1e-5, 1e-3)), ("batch", "choice", [128, 256])],
    "lstm": [("lr", "log", (3e-4, 3e-3)), ("hidden", "choice", [32, 48, 64, 96]), ("layers", "choice", [1, 2]),
             ("dropout", "lin", (0.0, 0.3)), ("weight_decay", "log", (1e-5, 1e-3)), ("batch", "choice", [128, 256])],
    "svm_nystroem": [("C", "log", (0.03, 30.0)), ("gamma", "log", (0.05, 4.0)), ("n_components", "choice", [1000, 2000, 4000])],
    "svm_rbf_exact": [("C", "log", (0.03, 30.0)), ("gamma", "log", (0.05, 4.0))],
    "svm_linear": [("C", "log", (0.001, 10.0))],
    "hgb": [("learning_rate", "log", (0.02, 0.2)), ("max_leaf_nodes", "choice", [15, 31, 63, 127]), ("min_samples_leaf", "int_log", (10, 80)),
            ("l2_regularization", "lin", (0.0, 2.0)), ("max_iter", "choice", [2000])],
}
FIXED = {
    "gru": dict(epochs=400, patience=25, clip=1.0, emb=48, head=96, optimizer="AdamW"),
    "lstm": dict(epochs=400, patience=25, clip=1.0, emb=48, head=96, optimizer="AdamW"),
    "svm_nystroem": dict(standardize=True, kernel="rbf (Nystroem approximation)", calibration="Platt on calibration matches"),
    "svm_rbf_exact": dict(standardize=True, kernel="rbf (exact, sklearn SVC)", max_train_rows=6000, calibration="Platt on calibration matches"),
    "svm_linear": dict(standardize=True, kernel="linear", calibration="Platt on calibration matches"),
    "hgb": dict(early_stopping="best iteration on early-stop matches (staged predictions)"),
}


def _sobol(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(max(n, 2))))
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]


def family_seed(family: str, salt: str = "m11-fair-search-v1") -> int:
    return int(hashlib.sha256(f"{salt}|{family}".encode()).hexdigest()[:8], 16)


def sample(family: str, n: int = N_CONFIGS, salt: str = "m11-fair-search-v1") -> list[dict[str, Any]]:
    space = SPACES[family]
    U = _sobol(n, len(space), family_seed(family, salt))
    out = []
    for k, u in enumerate(U):
        cfg: dict[str, Any] = {}
        for (name, kind, spec), v in zip(space, u):
            if kind == "log":
                cfg[name] = float(np.exp(np.log(spec[0]) + v * (np.log(spec[1]) - np.log(spec[0]))))
            elif kind == "lin":
                cfg[name] = float(spec[0] + v * (spec[1] - spec[0]))
            elif kind == "int_log":
                cfg[name] = int(round(np.exp(np.log(spec[0]) + v * (np.log(spec[1]) - np.log(spec[0])))))
            else:
                cfg[name] = spec[min(int(v * len(spec)), len(spec) - 1)]
        out.append(dict(config_id=f"{family}-{k:02d}", **cfg, **FIXED[family]))
    return out


def edge_flags(family: str, cfg: dict[str, Any], tol: float = 0.1) -> list[str]:
    """Parameters of a selected configuration that sit near a search boundary (log-space fraction ``tol``)."""
    flags = []
    for name, kind, spec in SPACES[family]:
        v = cfg[name]
        if kind in ("log", "int_log"):
            f = (np.log(v) - np.log(spec[0])) / (np.log(spec[1]) - np.log(spec[0]))
        elif kind == "lin":
            f = (v - spec[0]) / (spec[1] - spec[0])
        else:
            if len(spec) > 1 and v in (spec[0], spec[-1]) and isinstance(v, (int, float)):
                flags.append(f"{name}={v} is an end of the list")
            continue
        if f <= tol or f >= 1 - tol:
            flags.append(f"{name}={v:.4g} within {int(tol * 100)} % of the range edge")
    return flags
