"""Training distributions for the fair benchmark (train-only; evaluation stress tests are separate and fixed).

A distribution says which *variants* of each training encounter may be drawn and which online perturbations are applied:

* variants come from existing, verified tables of the same candidates: Stage 3 ``g0p40`` / ``g0p80`` / ``g1p60`` / ``g0p80_shift``
  and the Stage 4C multi-shift set (``shiftaug``: +-0.1, +-0.2, +-0.3 s). Only candidates of the primary ``g0p80`` table are used, so
  every distribution trains on the same encounters;
* **exactly one row per encounter per epoch** (Stage 4C rule): with probability ``p_centred`` the centred g0p80 row, otherwise one of
  the encounter's variants uniformly. Variants are augmentations, never extra independent data;
* online perturbations (training batches only): Gaussian position noise sigma drawn per batch from ``noise_sigma_m`` and random
  observation dropout with a rate drawn from ``drop_rates`` (the point nearest the gap of every tracklet is kept);
* ``visible_only`` is a control: training restricted to encounters whose hidden interval is both-VISIBLE (status is used to pick
  training rows for this reference model only; it is never a model input).

Early-stopping, calibration and test rows are always the centred ``g0p80`` rows of their matches (one evaluation definition).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from football_fatigue.m11_benchmark import data as D

PRIMARY = "g0p80"
VARIANT_TABLES = ("g0p40", "g1p60", "g0p80_shift", "shiftaug")
EVAL_TABLES = ("g0p80", "g0p80_shift", "g0p40", "g1p60")          # stress tests reported for every model, never trained on by default


@dataclass(frozen=True)
class TrainDist:
    name: str
    variants: tuple[str, ...] = ()
    p_centred: float = 1.0
    noise_sigma_m: tuple[float, ...] = (0.0,)
    drop_rates: tuple[float, ...] = (0.0,)
    visible_only: bool = False
    note: str = ""

    def as_dict(self):
        return asdict(self)

    @property
    def online(self) -> bool:
        return any(s > 0 for s in self.noise_sigma_m) or any(r > 0 for r in self.drop_rates)


DISTRIBUTIONS = {
    "centred": TrainDist("centred", note="historical Stage 4 distribution (gap centred on closest approach, 0.8 s)"),
    "shift_mix": TrainDist("shift_mix", ("shiftaug", "g0p80_shift"), 0.5, note="Stage 4C config C equivalent (50 % centred, 50 % shifted)"),
    "gap_mix": TrainDist("gap_mix", ("g0p40", "g1p60"), 1 / 3, note="variable gap lengths 0.4 / 0.8 / 1.6 s"),
    "robust_mix": TrainDist("robust_mix", ("shiftaug", "g0p80_shift", "g0p40", "g1p60"), 0.4, (0.0, 0.0, 0.02, 0.05), (0.0, 0.0, 0.1, 0.3),
                            note="shifted + variable gaps + noise + missing observations"),
    "visible_only": TrainDist("visible_only", visible_only=True, note="reference control: both-VISIBLE hidden intervals only"),
}


@dataclass
class Pool:
    """Primary rows (index 0..n0-1) followed by variant rows; ``owner[r]`` is the primary row of variant row r - n0."""
    y: np.ndarray
    groups: np.ndarray
    gap: np.ndarray
    n0: int
    si: np.ndarray | None = None
    so: np.ndarray | None = None
    F: np.ndarray | None = None
    k_perm: int | None = None
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    variant_table: np.ndarray = field(default_factory=lambda: np.zeros(0, object))
    meta: Any = None
    V: np.ndarray | None = None             # (n0, max variants) variant row offsets, -1 padded
    n_var: np.ndarray | None = None

    def index(self) -> None:
        self.n_var = np.bincount(self.owner, minlength=self.n0) if len(self.owner) else np.zeros(self.n0, int)
        self.V = np.full((self.n0, max(1, int(self.n_var.max()) if len(self.n_var) else 1)), -1)
        fill = np.zeros(self.n0, int)
        for r, o in enumerate(self.owner):
            self.V[o, fill[o]] = r
            fill[o] += 1

    def eligible(self, rows: np.ndarray, dist: TrainDist) -> np.ndarray:
        rows = np.asarray(rows)
        if dist.visible_only:
            st = self.meta["status_in_gap"].astype(str).to_numpy()
            rows = rows[st[rows] == "both_fully_visible"]
        return rows

    def draw(self, rows: np.ndarray, dist: TrainDist, rng: np.random.Generator) -> np.ndarray:
        """One pool row per encounter in ``rows`` (primary indices)."""
        rows = np.asarray(rows)
        if not dist.variants or not len(self.owner):
            return rows
        allowed = np.isin(self.variant_table, list(dist.variants))
        nv = np.array([int(allowed[self.V[r][self.V[r] >= 0]].sum()) for r in rows]) if len(rows) else np.zeros(0, int)
        use_var = (rng.random(len(rows)) >= dist.p_centred) & (nv > 0)
        out = rows.copy()
        for i in np.flatnonzero(use_var):
            cand = self.V[rows[i]][self.V[rows[i]] >= 0]
            cand = cand[allowed[cand]]
            out[i] = self.n0 + int(cand[rng.integers(0, len(cand))])
        return out


def _cid(meta) -> np.ndarray:
    return meta["candidate_id"].astype(str).to_numpy()


def load_pool(dist: TrainDist, *, sequences: bool, features: bool) -> Pool:
    """Primary g0p80 plus the variant tables ``dist`` can draw (guarded loaders; held-out refused)."""
    from football_fatigue.m11_phase2 import common as C
    x, y, g = D.load_inputs(PRIMARY)
    meta = D.load_meta(PRIMARY)
    C.checked(sorted({str(v) for v in np.asarray(g)}))
    pool = Pool(y=y.astype(int), groups=np.asarray(g).astype(str), gap=x["gap_s"].astype(np.float32), n0=len(y), meta=meta)
    if sequences:
        pool.si, pool.so = D.seq_tensor(x)
    if features:
        d = D.load_npz(D.out_dir() / "features" / f"{PRIMARY}.npz")
        pool.F, pool.k_perm = d["F"], D.n_perm([str(v) for v in d["names"]])
    idx0 = {c: i for i, c in enumerate(_cid(meta))}
    owners, tabs, si, so, F, gap, yy = [], [], [], [], [], [], []
    for t in dist.variants:
        if t == "shiftaug":
            from football_fatigue.m11_benchmark import shiftaug as SA
            xv, yv, gv, mv = SA.load()
            fpath = C.PROC / "m11-stage4c-robustness-v1" / "features" / "shiftaug.npz"
        else:
            xv, yv, gv = D.load_inputs(t)
            mv = D.load_meta(t)
            fpath = D.out_dir() / "features" / f"{t}.npz"
        keep = np.array([c in idx0 for c in _cid(mv)])
        own = np.array([idx0[c] for c in _cid(mv)[keep]])
        if not (pool.groups[own] == np.asarray(gv).astype(str)[keep]).all():
            raise RuntimeError(f"{t}: variant rows are not aligned with their centred encounters")
        owners.append(own)
        tabs.append(np.full(len(own), t, object))
        gap.append(xv["gap_s"][keep].astype(np.float32))
        yy.append(np.asarray(yv)[keep].astype(int))
        if sequences:
            a, b = D.seq_tensor({k: v[keep] for k, v in xv.items()})
            si.append(a)
            so.append(b)
        if features:
            fz = D.load_npz(fpath)
            if len(fz["F"]) != len(keep):
                raise RuntimeError(f"{t}: feature cache rows {len(fz['F'])} != inputs {len(keep)}")
            F.append(fz["F"][keep])
    if owners:
        pool.owner, pool.variant_table = np.concatenate(owners), np.concatenate(tabs)
        pool.gap = np.r_[pool.gap, np.concatenate(gap)]
        pool.y = np.r_[pool.y, np.concatenate(yy)]
        pool.groups = np.r_[pool.groups, pool.groups[pool.owner]]
        if sequences:
            pool.si, pool.so = np.concatenate([pool.si, *si]), np.concatenate([pool.so, *so])
        if features:
            pool.F = np.concatenate([pool.F, *F])
    pool.index()
    return pool


def perturb_batch(a, b, sigma_m: float, drop: float, gen) -> tuple[Any, Any]:
    """Online noise / observation dropout on (B, 2, T, 4) torch tensors (x/POS_SCALE, y/POS_SCALE, mask, dt); index 0 of every
    tracklet is the point nearest the gap (incoming reversed) and is never dropped."""
    import torch
    if sigma_m > 0:
        m = a[..., 2:3]
        a = a.clone()
        b = b.clone()
        a[..., :2] += torch.randn(a[..., :2].shape, generator=gen, device=a.device) * (sigma_m / D.POS_SCALE_M) * m
        b[..., :2] += torch.randn(b[..., :2].shape, generator=gen, device=b.device) * (sigma_m / D.POS_SCALE_M) * b[..., 2:3]
    if drop > 0:
        def dropit(z):
            keep = torch.rand(z.shape[:3], generator=gen, device=z.device) >= drop
            keep[..., 0] = True
            return z * keep[..., None].to(z.dtype)
        a, b = dropit(a), dropit(b)
    return a, b
