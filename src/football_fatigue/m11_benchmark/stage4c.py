"""Stage 4C (train/CV matches only): shift-robust training, longer training, texture augmentation, HGB/GRU ensembles.

``python -m football_fatigue.m11_benchmark.stage4c <phase>``; outputs under ``data/processed/m11-stage4c-robustness-v1`` (written
once, resumable per fold unit). Stage 2 / 3 / 4 outputs are only read. Predeclared design (no outer-fold tuning):

* Neural runs use the Stage 4B ``gru_gaplast`` architecture, optimiser, regularisation, batch, patience and early-stop matches. One
  run with a 160-epoch budget records the best early-stopping state reached within 80, 120 and 160 epochs; the main random stream
  is consumed exactly as in Stage 4B, so the 80-epoch state of config ``A`` must reproduce the Stage 4B predictions (checked).
* Training-distribution configs (gap-centre shift; augmentation draws use a separate random stream):
  ``A`` centred only; ``B`` each epoch every training encounter is drawn uniformly from {centred} + its valid shifted versions;
  ``C`` each epoch centred with probability 0.5, otherwise a uniformly drawn valid shifted version (centred if none);
  ``Anoise`` centred, Gaussian position noise with sigma ~ U(0, 5 cm) per sample and batch on valid points only;
  ``Asmooth`` centred, each sample replaced with probability 0.5 per batch by its mask-aware 5-point moving average.
  Early stopping uses the same distribution as training (B: equal weight per encounter over its versions; C: 0.5 centred + 0.5
  shifted; texture configs: original inputs).
* One draw per encounter per epoch: shifted versions never add encounters or weight.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import data as D
from . import folds as FO
from . import metrics as MT
from . import models as MD
from . import shiftaug as SA
from .robust import perturb, smooth
from .run import PRIMARY, Ctx, pdir, write_json_once

OUT = "m11-stage4c-robustness-v1"
CAPS = (80, 120, 160)
CONFIGS = ("A", "B", "C", "Anoise", "Asmooth")
KIND = "gru_gaplast"
AUG_SEED_OFFSET = 7919


def out_dir() -> Path:
    return D.proc_root() / OUT


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    p = out_dir() / "PROGRESS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- data bundle

class Bundle:
    """Centred primary table + shift-augmentation rows + evaluation tables, all aligned to the frozen outer folds."""

    def __init__(self, need_aug: bool = True):
        self.c = Ctx(PRIMARY, need_seq=True)
        self.fmap = self.c.fmap
        self.n0 = len(self.c.y)
        cid0 = self.c.meta["candidate_id"].to_numpy()
        self.idx0 = {c: i for i, c in enumerate(cid0)}
        if need_aug:
            xa, ya, ga, ma = SA.load()
            self.aug_meta = ma
            self.ya = ya.astype(int)
            self.ga = ga
            self.sia, self.soa = D.seq_tensor(xa)
            self.xa = xa
            owner = np.array([self.idx0[c] for c in ma["candidate_id"]])
            if not (self.c.g[owner] == ga).all():
                raise RuntimeError("augmentation rows are not aligned with their centred candidates")
            self.owner = owner
            self.fold_a = FO.fold_index(ga, self.fmap)
            n_aug = np.bincount(owner, minlength=self.n0)
            self.V = np.full((self.n0, max(1, n_aug.max())), -1)
            fill = np.zeros(self.n0, int)
            for r, o in enumerate(owner):
                self.V[o, fill[o]] = r
                fill[o] += 1
            self.n_aug = n_aug
        self.evals: dict[str, dict[str, Any]] = {}
        for t in ("g0p80_shift", "g0p40", "g1p60"):
            x, y, g = D.load_inputs(t)
            si, so = D.seq_tensor(x)
            self.evals[t] = dict(si=si, so=so, gap=x["gap_s"].astype(np.float32), y=y.astype(int), fold=FO.fold_index(g, self.fmap))
        x = self.c.x
        for vn, xv in (("smooth5", dict(x, in_pos=smooth(x["in_pos"], x["in_mask"]), out_pos=smooth(x["out_pos"], x["out_mask"]))),
                       ("noise5cm", dict(x, in_pos=perturb(x["in_pos"], x["in_mask"], 0.05, 1), out_pos=perturb(x["out_pos"], x["out_mask"], 0.05, 2)))):
            si, so = D.seq_tensor(xv)
            self.evals[vn] = dict(si=si, so=so, gap=self.c.gap.astype(np.float32), y=self.c.y, fold=self.c.fold)
        sm = dict(x, in_pos=smooth(x["in_pos"], x["in_mask"]), out_pos=smooth(x["out_pos"], x["out_mask"]))
        self.si_smooth, self.so_smooth = D.seq_tensor(sm)


# ---------------------------------------------------------------- trainer

def draw_rows(config: str, cands: np.ndarray, n_aug, V, n0: int, arng) -> np.ndarray:
    """Pool row per encounter for one epoch (centred rows are 0..n0-1, augmentation rows n0 + r; ``V[i]`` lists encounter i's
    augmentation rows, ``n_aug[i]`` how many). Exactly one row per encounter."""
    if config not in ("B", "C"):
        return cands
    na = n_aug[cands]
    u = arng.random(len(cands))
    if config == "B":
        k = np.floor(u * (1 + na)).astype(int)
        pick = V[cands, np.maximum(k - 1, 0)]
        return np.where(k == 0, cands, n0 + pick)
    coin = arng.random(len(cands)) < 0.5
    k = np.floor(u * np.maximum(na, 1)).astype(int)
    pick = V[cands, k]
    return np.where(coin & (na > 0), n0 + pick, cands)


def fit_caps(B: Bundle, config: str, seed: int, tr: np.ndarray, ev: np.ndarray, dev: str, pool: dict[str, Any]) -> dict[str, Any]:
    """Train one fold model; return the best early-stopping state (and epoch) within each cap and the history."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + AUG_SEED_OFFSET)
    net = MD.make_net(KIND, dropout=0.1, **MD.NEURAL[KIND]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    PSI, PSO, PG, PY = pool["si"], pool["so"], pool["gap"], pool["y"]
    tri = np.flatnonzero(tr)
    n0 = B.n0

    def draw(cands: np.ndarray) -> np.ndarray:
        return draw_rows(config, cands, B.n_aug if config in ("B", "C") else None, getattr(B, "V", None), n0, arng)

    # early-stopping rows and weights (same distribution as training)
    evi = np.flatnonzero(ev)
    if config in ("B", "C"):
        rows, w = [evi], []
        na = B.n_aug[evi]
        aug_rows = np.flatnonzero(np.isin(B.owner, evi))
        rows.append(n0 + aug_rows)
        if config == "B":
            w = np.r_[1.0 / (1 + na), 1.0 / (1 + B.n_aug[B.owner[aug_rows]])]
        else:
            w = np.r_[np.where(na > 0, 0.5, 1.0), 0.5 / B.n_aug[B.owner[aug_rows]]]
        es_rows, es_w = np.concatenate(rows), w
    else:
        es_rows, es_w = evi, np.ones(len(evi))
    es_t = torch.as_tensor(es_rows, device=dev)
    es_y = PY[es_rows]

    def predict_rows(si, so, g):
        net.eval()
        out = []
        with torch.no_grad():
            for s0 in range(0, len(g), 4096):
                out.append(torch.sigmoid(net(si[s0:s0 + 4096], so[s0:s0 + 4096], g[s0:s0 + 4096])).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    best, best_state, best_ep, bad, hist, snaps = np.inf, None, 0, 0, [], {}
    ytr_dev = torch.as_tensor(PY.astype(np.float32), device=dev)
    w1 = torch.ones(256, device=dev)                     # unit weights: the Stage 4B loss expression, kept for exact reproduction
    for ep in range(max(CAPS)):
        net.train()
        perm = rng.permutation(len(tri))
        src = torch.as_tensor(draw(tri), device=dev)
        tot = 0.0
        for s0 in range(0, len(perm), 256):
            idx = src[torch.as_tensor(perm[s0:s0 + 256], device=dev)]
            a, b, t, gg = PSI[idx].clone(), PSO[idx].clone(), ytr_dev[idx], PG[idx]
            if config == "Anoise":
                sig = torch.as_tensor(arng.uniform(0, 0.05, len(idx)) / D.POS_SCALE_M, device=dev, dtype=torch.float32)
                for z in (a, b):
                    nz = torch.as_tensor(arng.standard_normal((*z.shape[:-1], 2)), device=dev, dtype=torch.float32)
                    z[..., :2] += nz * sig[:, None, None, None] * z[..., 2:3]
            elif config == "Asmooth":
                use = torch.as_tensor(arng.random(len(idx)) < 0.5, device=dev)
                a = torch.where(use[:, None, None, None], pool["si_s"][idx], a)
                b = torch.where(use[:, None, None, None], pool["so_s"][idx], b)
            if rng.random() < 0.5:
                a[..., 1] *= -1
                b[..., 1] *= -1
            if rng.random() < 0.5:
                b, t = b.flip(1), 1 - t
            if rng.random() < 0.5:
                a, t = a.flip(1), 1 - t
            opt.zero_grad()
            ww = w1[: len(idx)]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(a, b, gg), t, reduction="none") * ww).sum() / ww.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        pv = predict_rows(PSI[es_t], PSO[es_t], PG[es_t])
        vll = float((MT.ll_vec(es_y, pv) * es_w).sum() / es_w.sum())
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(tri), val_log_loss=vll))
        if vll < best - 1e-4:
            best, bad, best_ep = vll, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        stop = bad >= 10
        for cap in CAPS:
            if cap not in snaps and (ep + 1 == cap or stop):
                snaps[cap] = dict(state=best_state, best_epoch=best_ep, es_loss=best, epochs_run=ep + 1)
        if stop:
            break
    net.eval()
    return dict(net=net, snaps=snaps, history=hist)


def predict(net, si, so, g, dev):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(g), 4096):
            f = lambda a: torch.as_tensor(np.ascontiguousarray(a[s0:s0 + 4096]), device=dev)  # noqa: E731
            out.append(torch.sigmoid(net(f(si), f(so), f(g))).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def run_config(B: Bundle, config: str, seeds) -> None:
    import torch
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = B.c
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
    if config in ("B", "C"):
        pool = dict(si=f32(np.concatenate([c.si, B.sia])), so=f32(np.concatenate([c.so, B.soa])),
                    gap=f32(np.r_[c.gap, B.xa["gap_s"]].astype(np.float32)), y=np.r_[c.y, B.ya])
    else:
        pool = dict(si=f32(c.si), so=f32(c.so), gap=f32(c.gap.astype(np.float32)), y=c.y)
    if config == "Asmooth":
        pool["si_s"], pool["so_s"] = f32(B.si_smooth), f32(B.so_smooth)
    for seed in seeds:
        d = out_dir() / "neural" / config / f"seed{seed}"
        if (d / "done.json").exists():
            log(f"{config} seed {seed}: done (skip)")
            continue
        t0 = time.perf_counter()
        for f in range(FO.N_OUTER):
            unit = d / f"fold{f}.npz"
            if unit.exists():
                continue
            t1 = time.perf_counter()
            es = set(FO.early_stop_matches(c.train_matches(f), f))
            ges = np.isin(c.g.astype(str), list(es))
            tr, ev, va = (c.fold != f) & ~ges, (c.fold != f) & ges, c.fold == f
            r = fit_caps(B, config, seed, tr, ev, dev, pool)
            arrays, info = {}, dict(fold=f, early_stop_matches=sorted(es, key=int), history=r["history"], caps={})
            for cap, sn in r["snaps"].items():
                r["net"].load_state_dict(sn["state"])
                net = r["net"]
                G = c.gap.astype(np.float32)
                arrays[f"c{cap}_val"] = predict(net, c.si[va], c.so[va], G[va], dev)
                arrays[f"c{cap}_es"] = predict(net, c.si[ev], c.so[ev], G[ev], dev)
                ptr = predict(net, c.si[tr], c.so[tr], G[tr], dev)
                if hasattr(B, "fold_a"):
                    m = B.fold_a == f
                    arrays[f"c{cap}_aug"] = predict(net, B.sia[m], B.soa[m], B.xa["gap_s"][m].astype(np.float32), dev)
                for t, e in B.evals.items():
                    m = e["fold"] == f
                    arrays[f"c{cap}_{t}"] = predict(net, e["si"][m], e["so"][m], e["gap"][m], dev)
                pv = arrays[f"c{cap}_val"]
                sym = float(np.abs(predict(net, c.si[va], c.so[va][:, ::-1].copy(), G[va], dev) - (1 - pv)).max())
                info["caps"][str(cap)] = dict(best_epoch=sn["best_epoch"], epochs_run=sn["epochs_run"], es_loss=sn["es_loss"],
                                              train=MT.basic(c.y[tr], ptr), val=MT.basic(c.y[va], pv), symmetry_max_err_outgoing=sym)
                d.mkdir(parents=True, exist_ok=True)
                torch.save(sn["state"], d / f"fold{f}_c{cap}.pt")
            info["fit_s"] = round(time.perf_counter() - t1, 1)
            D.write_once_npz(unit, arrays)
            write_json_once(unit.with_suffix(".json"), info)
            cs = info["caps"]
            log(f"  {config} s{seed} f{f}: " + "  ".join(f"c{k} BA {v['val']['balanced_accuracy']} LL {v['val']['log_loss']} ep {v['best_epoch']}"
                                                        for k, v in cs.items()) + f"  ({info['fit_s']} s)")
        write_json_once(d / "done.json", dict(config=config, seed=seed, runtime_s=round(time.perf_counter() - t0, 1)))
        log(f"{config} seed {seed}: finished ({time.perf_counter() - t0:.0f} s)")


def fill_aug(B: Bundle, config: str, seeds) -> None:
    """Shift-augmentation-set predictions for runs trained before that set existed (from the saved fold states; inference only)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        d = out_dir() / "neural" / config / f"seed{seed}"
        if not (d / "done.json").exists():
            continue
        for f in range(FO.N_OUTER):
            unit = d / f"fold{f}_aug.npz"
            if unit.exists() or "c80_aug" in D.load_npz(d / f"fold{f}.npz"):
                continue
            arr = {}
            m = B.fold_a == f
            for cap in CAPS:
                net = MD.make_net(KIND, dropout=0.1, **MD.NEURAL[KIND]).to(dev)
                net.load_state_dict(torch.load(d / f"fold{f}_c{cap}.pt", map_location=dev))
                arr[f"c{cap}_aug"] = predict(net, B.sia[m], B.soa[m], B.xa["gap_s"][m].astype(np.float32), dev)
            D.write_once_npz(unit, arr)
        log(f"fill_aug {config} seed {seed}: done")


def assemble(B: Bundle, config: str, seed: int, cap: int) -> dict[str, np.ndarray]:
    """Full out-of-fold prediction vectors of one (config, seed, cap)."""
    d = out_dir() / "neural" / config / f"seed{seed}"
    c = B.c
    out = dict(val=np.full(B.n0, np.nan), es=np.full(B.n0, np.nan))
    if hasattr(B, "fold_a"):
        out["aug"] = np.full(len(B.ya), np.nan)
    for t, e in B.evals.items():
        out[t] = np.full(len(e["y"]), np.nan)
    for f in range(FO.N_OUTER):
        z = D.load_npz(d / f"fold{f}.npz")
        out["val"][c.fold == f] = z[f"c{cap}_val"]
        info = json.loads((d / f"fold{f}.json").read_text(encoding="utf-8"))
        ev = np.isin(c.g.astype(str), info["early_stop_matches"])
        out.setdefault("es_by_fold", {})[f] = (np.flatnonzero(ev), z[f"c{cap}_es"])
        if "aug" in out:
            out["aug"][B.fold_a == f] = z[f"c{cap}_aug"]
        for t, e in B.evals.items():
            out[t][e["fold"] == f] = z[f"c{cap}_{t}"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["neural", "fill_aug", "hgb", "evaluate", "audit"])
    ap.add_argument("--configs", nargs="+", default=["A"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    if a.phase == "neural":
        need_aug = any(cf in ("B", "C") for cf in a.configs)
        B = Bundle(need_aug=need_aug)
        for cf in a.configs:
            run_config(B, cf, a.seeds)
    elif a.phase == "fill_aug":
        B = Bundle(need_aug=True)
        for cf in a.configs:
            fill_aug(B, cf, a.seeds)
    elif a.phase == "hgb":
        from . import stage4c_eval as EV
        EV.run_hgb()
    elif a.phase == "evaluate":
        from . import stage4c_eval as EV
        EV.evaluate()
    elif a.phase == "audit":
        from . import stage4c_eval as EV
        EV.audit()
    log(f"stage4c {a.phase} {a.configs} {a.seeds} finished in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
