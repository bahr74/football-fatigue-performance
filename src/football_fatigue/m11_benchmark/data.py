"""Data access for Stage 4: model inputs (through the Stage 3 adapter only), evaluation metadata (separate path, never a model
input), the conversion into the ``m11_recovery`` sample format and the motion features.

Probabilities in Stage 4 are always p = P(SWAPPED) (Stage 3 convention: y = 0 MATCHED, y = 1 SWAPPED). ``m11_recovery`` models and
features use the opposite convention (y = 1 = matched, positive margins favour matched); every crossing of that boundary goes
through ``samples.stage3_y_to_recovery`` / ``recovery_y_to_stage3`` or an explicit sign flip documented here.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_recovery import motion as MO
from football_fatigue.m11_recovery.views import Sample
from football_fatigue.window_intersections import local as L
from football_fatigue.window_intersections import samples as S

STAGE3 = "m11-stage3-samples-v1"
OUT = "m11-stage4-benchmarks-v1"
POS_SCALE_M = 10.0                       # fixed (not data-fitted) position scale for neural inputs


def proc_root() -> Path:
    from football_fatigue.window_intersections.__main__ import PROC
    return PROC


def stage3_dir() -> Path:
    return proc_root() / STAGE3


def out_dir() -> Path:
    return proc_root() / OUT


def load_inputs(table: str, matches: list[str] | None = None):
    """(inputs, y, groups) through the Stage 3 adapter (refuses held-out matches, verifies hashes)."""
    return S.load_model_inputs(stage3_dir(), table, matches)


def load_meta(table: str, matches: list[str] | None = None) -> pd.DataFrame:
    """Evaluation metadata, row-aligned with ``load_inputs`` (same match order). Train/CV matches only. Never a model input."""
    from football_fatigue.window_intersections.split import load_split

    sp = load_split(proc_root(), S.SPLIT)
    root = stage3_dir() / "matches"
    avail = sorted((p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith(".partial")), key=int)
    matches = avail if matches is None else [str(m) for m in matches]
    if set(matches) & set(sp["heldout"]) or set(avail) & set(sp["heldout"]):
        raise S.HeldOutRefused("held-out metadata is not readable during model development")
    parts = []
    for m in matches:
        L.verify_match_dir(root / m)
        parts.append(pd.read_parquet(root / m / f"meta_{table}.parquet"))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- conversion to m11_recovery samples (inputs only, never y)

def to_sample(x: dict[str, np.ndarray], i: int) -> Sample:
    """Sample i of an input table as a ``views.Sample``: NaN where masked, times relative to the gap centre. The label slot is a
    dummy (0): the label never passes through this path."""
    g = float(x["gap_s"][i])
    b = np.where(x["in_mask"][i][..., None], x["in_pos"][i], np.nan).astype(np.float64)
    a = np.where(x["out_mask"][i][..., None], x["out_pos"][i], np.nan).astype(np.float64)
    return Sample(b, a, x["in_dt"][i].astype(np.float64) - g / 2, x["out_dt"][i].astype(np.float64) + g / 2, g, 0.0, 0, ("", i))


def features(x: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Motion features (``motion.features``; perm_* first) and the 8 baseline margins for every sample.

    Sign convention: ``motion`` margins and perm_* features are cost(crossed) - cost(straight), positive favouring MATCHED.
    They are returned unchanged; models handle the sign through the label."""
    n = len(x["gap_s"])
    F, Mg = [], np.zeros((n, len(MO.BASELINES)))
    for i in range(n):
        s = to_sample(x, i)
        k = MO.Kin(s)
        pair, ctx = MO.pair_matrices(s, k)
        F.append(MO.features(s, pair, ctx))
        Mg[i] = [MO.margin(pair[f"d_{b}"]) for b in MO.BASELINES]
    return np.asarray(F), Mg, list(MO.FEATURE_NAMES)


def n_perm(names: list[str]) -> int:
    k = sum(nm.startswith("perm_") for nm in names)
    if names[:k] != [nm for nm in names if nm.startswith("perm_")]:
        raise ValueError("perm_* features must come first")
    return k


def swap_features(X: np.ndarray, k: int) -> np.ndarray:
    """Features of the same sample with the outgoing order exchanged: antisymmetric perm_* negate, context is invariant."""
    Z = X.copy()
    Z[:, :k] *= -1
    return Z


def seq_tensor(x: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Neural inputs (N, 2, T, 4) for incoming (reversed: index 0 nearest the gap) and outgoing tracklets: x/POS_SCALE,
    y/POS_SCALE, mask, dt (s). Masked steps are all-zero except the mask channel (= 0)."""
    def build(pos, mask, dt, reverse):
        n = len(pos)
        t = np.broadcast_to(dt[:, None, :], mask.shape)
        z = np.concatenate([pos * mask[..., None] / POS_SCALE_M, mask[..., None].astype(np.float32), (t * mask)[..., None]], -1).astype(np.float32)
        return z[:, :, ::-1].copy() if reverse else z
    return (build(x["in_pos"], x["in_mask"], x["in_dt"], True), build(x["out_pos"], x["out_mask"], x["out_dt"], False))


def sha_arrays(**arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        b = io.BytesIO()
        np.save(b, np.ascontiguousarray(arrs[k]), allow_pickle=False)
        h.update(k.encode())
        h.update(b.getvalue())
    return h.hexdigest()


def write_once_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write a deterministic npz once (refuse to overwrite a different file; identical content is a no-op). Returns sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    S.save_npz(tmp, arrays)
    new = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if path.exists():
        old = hashlib.sha256(path.read_bytes()).hexdigest()
        tmp.unlink()
        if old != new:
            raise RuntimeError(f"{path} exists with different content; outputs are never overwritten")
        return old
    tmp.replace(path)
    return new


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def feature_table(table: str, log=print) -> dict[str, Any]:
    """Cached motion features of one table (derived from model inputs only; written once)."""
    p = out_dir() / "features" / f"{table}.npz"
    if p.exists():
        d = load_npz(p)
        return dict(F=d["F"], margins=d["margins"], groups=d["groups"], names=[str(v) for v in d["names"]])
    x, _y, g = load_inputs(table)
    import time
    t = time.perf_counter()
    F, Mg, names = features(x)
    log(f"features {table}: {F.shape} in {time.perf_counter() - t:.0f} s")
    write_once_npz(p, dict(F=F, margins=Mg, groups=g, names=np.array(names)))
    return dict(F=F, margins=Mg, groups=g, names=names)
