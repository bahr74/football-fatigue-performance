"""Model fitting for the fair benchmark: configurable neural pair nets, SVM variants (Nystroem, exact RBF on a subsample, linear),
HGB with match-disjoint early stopping, calibration on separate matches, convergence diagnostics and paired match bootstrap.

Roles (per outer fold, all match-disjoint): fit (parameters), early_stop (epochs / HGB iterations / config selection),
calibration (temperature / Platt and abstention thresholds), test (reported only; never used for any choice)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import metrics as MT

from .nets import NetConfig, make_pair_net, n_params

AT_CAP_WINDOW = 10


# ---------------------------------------------------------------- shared helpers

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Scale a (symmetric) score z so that sigmoid(a z) minimises log loss on the calibration rows (no intercept: keeps the
    MATCHED / SWAPPED symmetry). Returns a."""
    from scipy.optimize import minimize_scalar
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(a):
        q = 1 / (1 + np.exp(-a * z))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(1e-3, 50.0), method="bounded").x)


def slope(v) -> float:
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def paired_ba(y, pa, pb, groups, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Balanced-accuracy difference A - B with a match-level bootstrap (decisions only: works for margins or probabilities
    given as P(SWAPPED) >= 0.5)."""
    y, ga = np.asarray(y), np.asarray(groups).astype(str)
    da, db = np.asarray(pa) >= 0.5, np.asarray(pb) >= 0.5

    def ba(m):
        yy = y[m]
        return 0.5 * ((da[m][yy == 1]).mean() + (~da[m][yy == 0]).mean()) - 0.5 * ((db[m][yy == 1]).mean() + (~db[m][yy == 0]).mean())
    ms = np.unique(ga)
    idx = {m: np.flatnonzero(ga == m) for m in ms}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ms, len(ms))
        bs.append(ba(np.concatenate([idx[m] for m in pick])))
    return dict(diff=round(float(ba(np.ones(len(y), bool))), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                n_matches=int(len(ms)))


def convergence(hist: list[dict[str, Any]], cap: int, patience: int) -> dict[str, Any]:
    """Convergence and overfitting diagnostics from the per-epoch history (train loss, early-stop log loss)."""
    if not hist:
        return dict(best_epoch=None, epochs_run=0, epoch_cap=cap)
    es = np.array([h["es_log_loss"] for h in hist])
    tr = np.array([h["train_loss"] for h in hist])
    b = int(np.argmin(es))
    last = es[-AT_CAP_WINDOW:]
    return dict(best_epoch=b + 1, epochs_run=len(hist), epoch_cap=cap, at_cap=b + 1 >= cap - AT_CAP_WINDOW,
                stopped_by_patience=len(hist) < cap and len(hist) - (b + 1) >= patience,
                es_ll_best=round(float(es[b]), 5), es_ll_final=round(float(es[-1]), 5), es_rise_after_best=round(float(es[-1] - es[b]), 5),
                train_ll_at_best=round(float(tr[b]), 5), generalisation_gap_at_best=round(float(es[b] - tr[b]), 5),
                es_ll_slope_last10=slope(last), train_ll_slope_last10=slope(tr[-AT_CAP_WINDOW:]),
                overfitting=bool(es[-1] - es[b] > 0.01 and slope(tr[-AT_CAP_WINDOW:]) < 0),
                still_improving=bool(b + 1 >= len(hist) - 2 and slope(last) < -1e-4))


def save_checkpoint(net, path, meta: dict[str, Any]) -> dict[str, Any]:
    """Write-once checkpoint plus a JSON sidecar with its SHA-256, parameter count and metadata."""
    import hashlib
    from pathlib import Path

    import torch
    p = Path(path)
    if p.exists() or p.with_suffix(".json").exists():
        raise FileExistsError(f"{p} exists; checkpoints are never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    side = dict(meta, checkpoint=p.name, sha256=sha, params=n_params(net), keys=len(net.state_dict()))
    import json
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str), encoding="utf-8")
    return side


def verify_checkpoint(path, net_cfg: NetConfig) -> bool:
    """SHA-256 matches the sidecar and the weights load into a freshly built network of ``net_cfg``."""
    import hashlib
    import json
    from pathlib import Path

    import torch
    p = Path(path)
    side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(p.read_bytes()).hexdigest() != side["sha256"]:
        return False
    net = make_pair_net(net_cfg)
    net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return n_params(net) == side["params"]


# ---------------------------------------------------------------- neural

class FairNeural:
    """Historical training loop (AdamW, BCE, gradient clipping, reflection + incoming / outgoing exchange augmentation, best
    early-stop state) with every size and optimisation setting configurable and full convergence diagnostics."""

    def __init__(self, net: NetConfig, *, lr=1e-3, weight_decay=1e-4, batch=256, epochs=80, patience=10, clip=1.0, seed=0, device=None):
        self.cfg, self.lr, self.wd, self.batch, self.epochs, self.patience, self.clip, self.seed = net, lr, weight_decay, batch, epochs, patience, clip, seed
        self.device = device

    def fit(self, SI, SO, G, y, SIe, SOe, Ge, ye, log=None, *, draw=None, perturb=None, state_path=None,
            stop_at_epoch: int | None = None) -> "FairNeural":
        """Train with early stopping on (SIe, SOe, Ge, ye).

        ``draw(rng)`` returns the row indices of this epoch (one per encounter; default: every row). ``perturb`` = (sigmas, drops)
        draws a noise sigma and a dropout rate per batch (training batches only). ``state_path``: the full training state is saved
        after every epoch and training resumes from it (bit-identical on CPU; tested). ``stop_at_epoch``: pause after that epoch
        (successive halving); ``finished`` tells whether patience or the epoch cap ended training. Without draw / perturb /
        state_path the random stream is exactly the historical one."""
        import os

        import psutil
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        gen = torch.Generator().manual_seed(self.seed + 7919)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.net = make_pair_net(self.cfg).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
        si, so, g, yy = f(SI), f(SO), f(G.astype(np.float32)), f(y.astype(np.float32))
        best, best_state, bad, hist, start = np.inf, None, 0, [], 0
        proc = psutil.Process()
        peak_rss = proc.memory_info().rss
        if state_path is not None and os.path.exists(state_path):
            st = torch.load(state_path, map_location=dev, weights_only=False)
            self.net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            best, best_state, bad, hist, start = st["best"], st["best_state"], st["bad"], st["hist"], st["epoch"]
            rng.bit_generator.state = st["np_rng"]
            torch.set_rng_state(st["torch_rng"])
            gen.set_state(st["gen"])
            if st.get("finished"):
                start = self.epochs                      # nothing left to train
        finished = start >= self.epochs
        t0 = time.perf_counter() - (hist[-1]["elapsed_s"] if hist and "elapsed_s" in hist[-1] else 0.0)
        for ep in range(start, self.epochs):
            self.net.train()
            rows = np.arange(len(y)) if draw is None else np.asarray(draw(rng))
            perm = rows[rng.permutation(len(rows))]
            tot, nseen = 0.0, 0
            for s0 in range(0, len(perm), self.batch):
                idx = f(perm[s0:s0 + self.batch])
                a, b, t, gg = si[idx].clone(), so[idx].clone(), yy[idx], g[idx]
                if perturb is not None:
                    from .distributions import perturb_batch
                    sig = float(perturb[0][rng.integers(0, len(perturb[0]))])
                    dr = float(perturb[1][rng.integers(0, len(perturb[1]))])
                    if sig > 0 or dr > 0:
                        a, b = perturb_batch(a.cpu(), b.cpu(), sig, dr, gen)
                        a, b = a.to(dev), b.to(dev)
                if rng.random() < 0.5:
                    a[..., 1] *= -1
                    b[..., 1] *= -1
                if rng.random() < 0.5:
                    b, t = b.flip(1), 1 - t
                if rng.random() < 0.5:
                    a, t = a.flip(1), 1 - t
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(a, b, gg), t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                opt.step()
                tot += float(loss.detach()) * len(idx)
                nseen += len(idx)
            if not np.isfinite(tot):
                raise FloatingPointError(f"non-finite training loss at epoch {ep + 1}")
            pe = self.predict(SIe, SOe, Ge)
            vll = float(MT.ll_vec(ye, pe).mean())
            peak_rss = max(peak_rss, proc.memory_info().rss)
            hist.append(dict(epoch=ep + 1, train_loss=tot / max(nseen, 1), es_log_loss=vll, elapsed_s=round(time.perf_counter() - t0, 3)))
            if log:
                log(f"epoch {ep + 1}: train {tot / max(nseen, 1):.4f} es {vll:.4f}")
            if vll < best - 1e-4:
                best, bad = vll, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    finished = True
            if ep + 1 >= self.epochs:
                finished = True
            if state_path is not None:
                tmp = f"{state_path}.tmp"
                torch.save(dict(net=self.net.state_dict(), opt=opt.state_dict(), best=best, best_state=best_state, bad=bad, hist=hist,
                                epoch=ep + 1, np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), gen=gen.get_state(),
                                finished=finished), tmp)
                os.replace(tmp, state_path)
            if finished or (stop_at_epoch is not None and ep + 1 >= stop_at_epoch):
                break
        self.finished = finished
        self.history = hist
        self.best_es_log_loss = float(best)
        if finished and best_state is not None:
            self.net.load_state_dict(best_state)
        self.diagnostics = convergence(hist, self.epochs, self.patience)
        self.diagnostics.update(params=n_params(self.net), fit_s=round(time.perf_counter() - t0, 2), device=dev, finished=finished,
                                s_per_epoch=round(hist[-1]["elapsed_s"] / max(len(hist), 1), 3) if hist else None,
                                peak_rss_mb=round(peak_rss / 2**20, 1),
                                peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev == "cuda" else None)
        return self

    def logits(self, SI, SO, G) -> np.ndarray:
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for s0 in range(0, len(G), 4096):
                f = lambda a: torch.as_tensor(np.ascontiguousarray(a[s0:s0 + 4096]), device=dev)  # noqa: E731
                out.append(self.net(f(SI), f(SO), f(G.astype(np.float32))).cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def predict(self, SI, SO, G) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logits(SI, SO, G)))


# ---------------------------------------------------------------- tabular

def make_tabular(family: str, cfg: dict[str, Any], n_features: int, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    if family == "svm_nystroem":
        return make_pipeline(StandardScaler(), Nystroem(gamma=cfg["gamma"] / n_features, n_components=int(cfg["n_components"]), random_state=seed),
                             LinearSVC(C=cfg["C"], dual=False, max_iter=5000))
    if family == "svm_rbf_exact":
        return make_pipeline(StandardScaler(), SVC(C=cfg["C"], gamma=cfg["gamma"] / n_features, kernel="rbf", cache_size=1000))
    if family == "svm_linear":
        return make_pipeline(StandardScaler(), LinearSVC(C=cfg["C"], dual=False, max_iter=5000))
    if family == "hgb":
        return HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_leaf_nodes=int(cfg["max_leaf_nodes"]),
                                              min_samples_leaf=int(cfg["min_samples_leaf"]), l2_regularization=cfg["l2_regularization"],
                                              max_iter=int(cfg["max_iter"]), early_stopping=False, random_state=seed)
    raise ValueError(family)


class FairTabular:
    """Antisymmetric augmentation (fit on X and its outgoing-exchanged copy with flipped labels), symmetric score, scale fitted on
    calibration matches. ``score`` is a logit-like symmetric score (SVM decision value or HGB log-odds)."""

    def __init__(self, family: str, cfg: dict[str, Any], k_perm: int, seed: int = 0):
        self.family, self.cfg, self.k, self.seed, self.a = family, cfg, k_perm, seed, None
        self.best_iter: int | None = None
        self.diagnostics: dict[str, Any] = {}

    def _proba(self, X) -> np.ndarray:
        if self.best_iter is None or self.best_iter >= self.m.n_iter_:
            return self.m.predict_proba(X)[:, 1]
        for k, pp in enumerate(self.m.staged_predict_proba(X), 1):
            if k == self.best_iter:
                return pp[:, 1]
        raise RuntimeError("staged prediction ended early")

    def fit(self, X, y, Xe=None, ye=None, sample_weight=None):
        t0 = time.perf_counter()
        X = np.nan_to_num(X)
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        if self.family == "svm_rbf_exact":
            n = min(len(y), int(self.cfg.get("max_train_rows", 6000)))
            keep = np.sort(np.random.default_rng(self.seed).choice(len(y), n, replace=False))
            X, y = X[keep], y[keep]
            w = None if w is None else w[keep]
            self.diagnostics["train_rows_used"] = int(n)
        XX, yy = np.r_[X, D.swap_features(X, self.k)], np.r_[y, 1 - y]
        model = make_tabular(self.family, self.cfg, X.shape[1], self.seed)
        if w is None:
            self.m = model.fit(XX, yy)
        else:
            ww = np.r_[w, w]
            key = "sample_weight" if self.family == "hgb" else f"{model.steps[-1][0]}__sample_weight"
            self.m = model.fit(XX, yy, **{key: ww})
        if self.family == "hgb" and Xe is not None:
            Xe = np.nan_to_num(Xe)
            Ze = D.swap_features(Xe, self.k)
            lls = []
            for pa, pb in zip(self.m.staged_predict_proba(Xe), self.m.staged_predict_proba(Ze)):
                p = 0.5 * (pa[:, 1] + 1 - pb[:, 1])
                lls.append(float(MT.ll_vec(ye, p).mean()))
            best = int(np.argmin(lls) + 1)
            self.best_iter = best                      # predictions use the staged model at this iteration (no refit)
            self.diagnostics.update(best_iter=best, iter_cap=int(self.cfg["max_iter"]), at_cap=best >= int(self.cfg["max_iter"]) - 10)
        self.diagnostics["fit_s"] = round(time.perf_counter() - t0, 2)
        return self

    def score(self, X) -> np.ndarray:
        X = np.nan_to_num(X)
        Z = D.swap_features(X, self.k)
        if self.family.startswith("svm"):
            return 0.5 * (self.m.decision_function(X) - self.m.decision_function(Z))
        p = 0.5 * (self._proba(X) + 1 - self._proba(Z))
        return logit(p)

    def calibrate(self, Xc, yc) -> float:
        self.a = fit_temperature(self.score(Xc), yc)
        return self.a

    def predict(self, X) -> np.ndarray:
        z = self.score(X)
        a = 1.0 if self.a is None else self.a
        if self.a is None and self.family.startswith("svm"):
            raise RuntimeError("SVM scores need calibration matches before probabilities exist")
        return 1 / (1 + np.exp(-a * z))
