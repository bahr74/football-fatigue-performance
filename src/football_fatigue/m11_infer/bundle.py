"""Model bundles: the existing Stage 4C outer-fold checkpoints of GRU C and HGB C, with metadata and integrity checks.

Sources (read only, never rewritten): ``data/processed/m11-stage4c-robustness-v1``

- GRU C: ``neural/C/seed0/fold{f}_c160.pt`` (``gru_gaplast`` state dict; best early-stopping state within a 160-epoch budget) and
  its fold record ``fold{f}.json`` / stored out-of-fold predictions ``fold{f}.npz`` (key ``c160_val``).
- HGB C: ``hgb/C/fold{f}.joblib`` (a pickled ``m11_benchmark.models.Tabular``; symmetric prediction) with ``fold{f}.json`` /
  ``fold{f}.npz`` (key ``val``).

Fold ``f`` was trained without the four matches of outer fold ``f`` (``m11_benchmark.folds``); those matches are its unseen
matches. Joblib files are pickles: only the project's own files are loaded, after their SHA-256 has been recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import folds as FO
from football_fatigue.m11_benchmark import models as MD
from football_fatigue.window_intersections import samples as S

SOURCE = "m11-stage4c-robustness-v1"
OUT = "m11-infer-v1"
GRU_KIND = "gru_gaplast"
GRU_SEED = 0
GRU_CAP = 160
MODELS = ("gru_c", "hgb_c")

# Pooled development OOF thresholds on max(p, 1 - p) (independent subset, all 8 folds; GRU seed 0), Stage 4C report section 9.
# ``check-bundles`` recomputes them from the saved predictions and fails if they differ.
THRESHOLDS: dict[str, dict[str, float]] = {"gru_c": {"error_05": 0.6425, "error_02": 0.8525},
                                           "hgb_c": {"error_05": 0.7875, "error_02": 0.9029}}
OPERATING_POINTS = ("error_05", "error_02")

# Reproduction tolerances on |p_new - p_stored| (plus: no decision may flip).
# GRU: float32 recurrent network; the stored values were computed with CUDA kernels on batches. On CUDA with the same batching the
# reproduction is exact (measured max |dp| = 0.0); one sample at a time on CUDA uses other kernels (measured 3.0e-7), hence 1e-6.
# On CPU the float32 summation order differs (measured max 9.1e-5 on fold 0, 3.3e-5 on fold 4), so the CPU tolerance is 2.5e-4.
# HGB: float64 features recomputed by the same code; tree ensembles are deterministic (measured 0.0).
TOL = {"gru_c": {"cuda": 1e-6, "cpu": 2.5e-4}, "hgb_c": {"cuda": 1e-12, "cpu": 1e-12}}


def tolerance(name: str, device: str) -> float:
    return TOL[name]["cuda" if str(device).startswith("cuda") else "cpu"]


def proc_root() -> Path:
    return D.proc_root()


def source_dir() -> Path:
    return proc_root() / SOURCE


def out_dir() -> Path:
    return proc_root() / OUT


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_cv_matches() -> list[str]:
    """The 32 train/CV matches of the hash-verified frozen split."""
    from football_fatigue.window_intersections.split import load_split

    sp = load_split(proc_root(), S.SPLIT)
    if S.split_sha256(proc_root()) != S.SPLIT_SHA256:
        raise S.HeldOutRefused("the frozen split differs from the approved split")
    return sorted(sp["train_cv"], key=int)


def heldout_matches() -> list[str]:
    from football_fatigue.window_intersections.split import load_split

    return sorted(load_split(proc_root(), S.SPLIT)["heldout"], key=int)


def fold_map() -> dict[str, int]:
    return FO.outer_folds(train_cv_matches())


def fold_matches(fold: int) -> list[str]:
    """The unseen (outer validation) matches of fold ``fold``, in the Stage 4 row order (ascending match id)."""
    if fold not in range(FO.N_OUTER):
        raise ValueError(f"fold must be 0..{FO.N_OUTER - 1}")
    return sorted((m for m, f in fold_map().items() if f == fold), key=int)


def fold_for_match(match_id: str | int) -> int:
    """Outer fold whose model never saw ``match_id``. Held-out and unknown matches are refused."""
    m = str(int(match_id))
    if m in set(heldout_matches()):
        raise S.HeldOutRefused(f"{m} is a frozen held-out match; it is not usable before the approved final evaluation")
    fm = fold_map()
    if m not in fm:
        raise ValueError(f"{m} is not a train/CV match of the frozen split (official validation/test data are not used)")
    return fm[m]


@dataclass
class FoldModel:
    """One loaded outer-fold model. ``predict`` takes model-input arrays (``samples.INPUT_FIELDS``, batch first)."""

    name: str
    fold: int
    path: Path
    sha256: str
    meta: dict[str, Any]
    obj: Any = field(repr=False)
    device: str = "cpu"

    @property
    def unseen_matches(self) -> list[str]:
        return fold_matches(self.fold)

    def predict(self, x: dict[str, np.ndarray]) -> np.ndarray:
        if self.name == "gru_c":
            return predict_gru(self.obj, x, self.device)
        return predict_hgb(self.obj, x)


def predict_gru(net, x: dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
    import torch

    si, so = D.seq_tensor(x)
    g = x["gap_s"].astype(np.float32)
    net.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(g), 4096):
            f = lambda a: torch.as_tensor(np.ascontiguousarray(a[s0:s0 + 4096]), device=device)  # noqa: E731
            out.append(torch.sigmoid(net(f(si), f(so), f(g))).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def predict_hgb(model, x: dict[str, np.ndarray]) -> np.ndarray:
    if len(x["gap_s"]) == 0:
        return np.zeros(0)
    F, _margins, names = D.features(x)
    if D.n_perm(names) != model.k:
        raise RuntimeError("feature layout differs from the one the HGB model was trained on")
    return np.asarray(model.predict(F), np.float64)


def _fold_paths(name: str, fold: int) -> tuple[Path, Path, Path, str]:
    """(checkpoint, fold record json, stored predictions npz, stored OOF key)."""
    if fold not in range(FO.N_OUTER):
        raise ValueError(f"fold must be 0..{FO.N_OUTER - 1}")
    if name == "gru_c":
        d = source_dir() / "neural" / "C" / f"seed{GRU_SEED}"
        return d / f"fold{fold}_c{GRU_CAP}.pt", d / f"fold{fold}.json", d / f"fold{fold}.npz", f"c{GRU_CAP}_val"
    if name == "hgb_c":
        d = source_dir() / "hgb" / "C"
        return d / f"fold{fold}.joblib", d / f"fold{fold}.json", d / f"fold{fold}.npz", "val"
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")


def load_fold_model(name: str, fold: int, device: str = "cpu", expected_sha256: str | None = None) -> FoldModel:
    ckpt, rec, _npz, _key = _fold_paths(name, fold)
    for p in (ckpt, rec):
        if not p.exists():
            raise FileNotFoundError(p)
    sha = sha256_file(ckpt)
    if expected_sha256 is None:
        expected_sha256 = recorded_sha256(name, fold)
    if expected_sha256 is not None and sha != expected_sha256:
        raise RuntimeError(f"{ckpt} SHA-256 {sha} differs from the recorded {expected_sha256}")
    info = json.loads(rec.read_text(encoding="utf-8"))
    if int(info["fold"]) != fold:
        raise RuntimeError(f"{rec} describes fold {info['fold']}, not {fold}")
    meta: dict[str, Any] = dict(model=name, fold=fold, checkpoint=_rel(ckpt), checkpoint_sha256=sha,
                                early_stop_matches=info.get("early_stop_matches"), unseen_matches=fold_matches(fold))
    if name == "gru_c":
        import torch

        net = MD.make_net(GRU_KIND, dropout=0.1, **MD.NEURAL[GRU_KIND])
        state = torch.load(ckpt, map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.to(device).eval()
        cap = info["caps"][str(GRU_CAP)]
        meta.update(kind=GRU_KIND, config="C", seed=GRU_SEED, epoch_budget=GRU_CAP, best_epoch=cap["best_epoch"],
                    epochs_run=cap["epochs_run"], budget_binding=cap["epochs_run"] >= GRU_CAP, n_params=MD.n_params(net),
                    architecture=dict(MD.NEURAL[GRU_KIND], dropout=0.1, T=S.T, channels=["x/10 m", "y/10 m", "mask", "dt_s"]))
        obj = net
    else:
        import joblib

        obj = joblib.load(ckpt)
        if not isinstance(obj, MD.Tabular) or obj.name != "hgb":
            raise RuntimeError(f"{ckpt} is not an HGB Tabular model")
        meta.update(kind="hgb", config="C", hyperparameters=info.get("selected"), n_perm_features=int(obj.k))
    return FoldModel(name, fold, ckpt, sha, meta, obj, device)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(proc_root().parent.parent)).replace("\\", "/")
    except ValueError:
        return str(p)


def stored_oof(name: str, fold: int) -> np.ndarray:
    _c, _r, npz, key = _fold_paths(name, fold)
    return D.load_npz(npz)[key]


def manifest_path() -> Path:
    return out_dir() / "bundles_v1.json"


def recorded_sha256(name: str, fold: int) -> str | None:
    """Checkpoint hash recorded by ``check-bundles`` (None before the first check)."""
    p = manifest_path()
    if not p.exists():
        return None
    man = json.loads(p.read_text(encoding="utf-8"))
    return man["bundles"][name][str(fold)]["checkpoint_sha256"]


def fold_inputs(fold: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Centred 0.8 s Stage 3 inputs of the fold's unseen matches, in the stored OOF row order (via the Stage 3 adapter)."""
    return D.load_inputs(S.PRIMARY, fold_matches(fold))


def pooled_thresholds() -> dict[str, dict[str, float | None]]:
    """Recompute the pooled development thresholds from the saved OOF predictions (independent subset, all folds)."""
    from football_fatigue.m11_benchmark import metrics as MT

    x, y, g = D.load_inputs(S.PRIMARY)
    meta = D.load_meta(S.PRIMARY)
    indep = meta["independent"].to_numpy(bool)
    fm = fold_map()
    fold = FO.fold_index(g, fm)
    out: dict[str, dict[str, float | None]] = {}
    for name in MODELS:
        p = np.full(len(y), np.nan)
        for f in range(FO.N_OUTER):
            p[fold == f] = stored_oof(name, f)
        cc = MT.coverage_curve(y[indep].astype(int), p[indep])["coverage_at_error"]
        out[name] = {op: cc[op]["threshold"] for op in OPERATING_POINTS}
        out[name] |= {f"coverage_{op}": cc[op]["coverage"] for op in OPERATING_POINTS}
    return out


def write_json_once(path: Path, obj: Any) -> None:
    txt = json.dumps(obj, indent=1, sort_keys=True, default=_jsonable)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != txt:
            raise RuntimeError(f"{path} exists with different content; outputs are never overwritten")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v))


def check_bundles(device: str = "cpu", folds=range(FO.N_OUTER), log=print) -> dict[str, Any]:
    """Load every fold model, reproduce its stored out-of-fold predictions on the centred 0.8 s table, test outgoing-order
    symmetry and recompute the pooled thresholds. Returns the report (does not write)."""
    rep: dict[str, Any] = dict(schema="m11-infer-bundle-check-v1", source=SOURCE, device=device, tolerance=TOL, bundles={m: {} for m in MODELS})
    for f in folds:
        x, y, _g = fold_inputs(f)
        xs = dict(x, out_pos=x["out_pos"][:, ::-1].copy(), out_mask=x["out_mask"][:, ::-1].copy())
        for name in MODELS:
            fm = load_fold_model(name, f, device=device, expected_sha256=None)
            p = fm.predict(x)
            ref = stored_oof(name, f)
            if p.shape != ref.shape:
                raise RuntimeError(f"{name} fold {f}: {p.shape} predictions, {ref.shape} stored")
            d = float(np.abs(p - ref).max())
            flips = int(((p > 0.5) != (ref > 0.5)).sum())
            th = THRESHOLDS[name]["error_05"]
            abst_flips = int(((np.maximum(p, 1 - p) < th) != (np.maximum(ref, 1 - ref) < th)).sum())
            sym = float(np.abs(fm.predict(xs) - (1 - p)).max())
            r = dict(fm.meta, n=int(len(p)), max_abs_diff_vs_stored=d, tolerance=tolerance(name, device), decision_flips=flips,
                     abstention_flips_error_05=abst_flips, reproduced=d <= tolerance(name, device) and flips == 0,
                     symmetry_max_err=sym, balanced_accuracy=_ba(y, p))
            rep["bundles"][name][str(f)] = r
            log(f"{name} fold {f}: n {len(p)}  max|dp| {d:.3g}  flips {flips}/{abst_flips}  symmetry {sym:.3g}  BA {r['balanced_accuracy']}  "
                f"{'OK' if r['reproduced'] else 'MISMATCH'}")
    th = pooled_thresholds()
    rep["thresholds_recomputed"] = th
    rep["thresholds_used"] = THRESHOLDS
    rep["thresholds_match"] = all(th[m][op] == THRESHOLDS[m][op] for m in MODELS for op in OPERATING_POINTS)
    rep["all_reproduced"] = all(r["reproduced"] for v in rep["bundles"].values() for r in v.values())
    rep["code_sha256"] = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return rep


def _ba(y, p) -> float:
    pred = (p > 0.5).astype(int)
    y = np.asarray(y).astype(int)
    return round(float(np.mean([np.mean(pred[y == k] == k) for k in (0, 1)])), 4)
