"""Association models for two-player identity recovery. Every model returns P(H0) for a scenario table.

- :class:`Baseline`: one motion baseline's cost margin (cost(H1) - cost(H0)), turned into a probability by a logistic fit of
  ``margin / (0.25 + gap)`` on the training matches only;
- :class:`GBM`: histogram gradient boosting on the permutation features; trained on both after orders (antisymmetric
  augmentation) and predicted symmetrically, so swapping the anonymous after tracks gives exactly 1 - p;
- :class:`PairNet` (``hybrid``): GRU encoders of each before / after track plus the physics pair quantities give a pair score
  s(i, j); logit(H0) = s00 + s11 - s01 - s10, the one-to-one assignment score (generalises to N players with the Hungarian
  algorithm). Trained on the training matches with early stopping on one held-out training match.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .motion import BASELINES


def n_perm(names: list[str]) -> int:
    return sum(n.startswith("perm_") for n in names)


class Baseline:
    def __init__(self, name: str):
        self.name, self.k = name, BASELINES.index(name)

    def _x(self, t):
        return (t["margins"][:, self.k] / (0.25 + t["gap"]))[:, None]

    def fit(self, t):
        self.lr = LogisticRegression(C=1e3, fit_intercept=False).fit(np.r_[self._x(t), -self._x(t)], np.r_[t["y"], 1 - t["y"]])
        return self

    def predict(self, t):
        return self.lr.predict_proba(self._x(t))[:, 1]


class GBM:
    def __init__(self, names: list[str], seed: int = 0, **kw):
        self.np = n_perm(names)
        self.params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                           random_state=seed) | kw

    def _swap(self, X):
        Z = X.copy()
        Z[:, :self.np] *= -1
        return Z

    def fit(self, t):
        X = np.nan_to_num(t["feats"], nan=0.0)
        self.m = HistGradientBoostingClassifier(**self.params).fit(np.r_[X, self._swap(X)], np.r_[t["y"], 1 - t["y"]])
        return self

    def predict(self, t):
        X = np.nan_to_num(t["feats"], nan=0.0)
        return 0.5 * (self.m.predict_proba(X)[:, 1] + 1 - self.m.predict_proba(self._swap(X))[:, 1])


# ---------------------------------------------------------------- hybrid sequence model

def _torch():
    import torch

    return torch


def make_pairnet(n_phys: int, hidden: int = 64):
    torch = _torch()
    nn = torch.nn

    class PairNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_b = nn.GRU(4, hidden, batch_first=True)
            self.enc_a = nn.GRU(4, hidden, batch_first=True)
            self.pair = nn.Sequential(nn.Linear(3 * hidden + n_phys + 1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def encode(self, enc, seq):                        # seq (B, 2, T, 4) -> (B, 2, H)
            B = seq.shape[0]
            _, h = enc(seq.reshape(B * 2, *seq.shape[2:]))
            return h[-1].reshape(B, 2, -1)

        def forward(self, bseq, aseq, phys, gap):
            eb, ea = self.encode(self.enc_b, bseq), self.encode(self.enc_a, aseq)
            s = []
            for i in (0, 1):
                for j in (0, 1):
                    z = torch.cat([eb[:, i], ea[:, j], eb[:, i] * ea[:, j], phys[:, i, j], gap[:, None]], 1)
                    s.append(self.pair(z)[:, 0])
            s00, s01, s10, s11 = s
            return s00 + s11 - s01 - s10

    return PairNet()


class Hybrid:
    def __init__(self, seed: int = 0, epochs: int = 30, batch: int = 512, lr: float = 1e-3, patience: int = 5, hidden: int = 64):
        self.seed, self.epochs, self.batch, self.lr, self.patience, self.hidden = seed, epochs, batch, lr, patience, hidden

    def _tensors(self, t, dev):
        torch = _torch()
        phys = np.nan_to_num(t["phys"], nan=0.0).astype(np.float32)
        phys = (phys - self.mu) / self.sd
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
        return f(t["bseq"].astype(np.float32)), f(t["aseq"].astype(np.float32)), f(phys), f(t["gap"].astype(np.float32))

    def fit(self, t, val=None):
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        phys = np.nan_to_num(t["phys"], nan=0.0)
        self.mu, self.sd = phys.mean((0, 1, 2)).astype(np.float32), (phys.std((0, 1, 2)) + 1e-6).astype(np.float32)
        self.net = make_pairnet(t["phys"].shape[-1], self.hidden).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        B, A, P, G = self._tensors(t, dev)
        Y = torch.as_tensor(t["y"].astype(np.float32), device=dev)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(Y))
            for s in range(0, len(perm), self.batch):
                idx = torch.as_tensor(perm[s:s + self.batch], device=dev)
                b, a = B[idx].clone(), A[idx].clone()
                if rng.random() < 0.5:                       # reflection augmentation (y -> -y); physics features are invariant
                    b[..., 1] *= -1
                    a[..., 1] *= -1
                p, y = P[idx], Y[idx]
                if rng.random() < 0.5:                       # exchange the after order: label flips, pair features transpose
                    a, p, y = a.flip(1), p.flip(2), 1 - y
                opt.zero_grad()
                loss = lossf(self.net(b, a, p, G[idx]), y)
                loss.backward()
                opt.step()
            if val is not None:
                pv = self.predict(val)
                yv = val["y"]
                ll = float(-np.mean(yv * np.log(np.clip(pv, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - pv, 1e-6, 1))))
                self.history.append(ll)
                if ll < best - 1e-4:
                    best, bad = ll, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.epochs_run = len(self.history) if val is not None else self.epochs
        return self

    def predict(self, t):
        torch = _torch()
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(t["gap"]), 4096):
                sub = {k: v[s:s + 4096] for k, v in t.items()}
                B, A, P, G = self._tensors(sub, dev)
                z1 = self.net(B, A, P, G)
                z2 = self.net(B, A.flip(1), P.flip(2), G)      # symmetrised over the after order
                out.append(torch.sigmoid(0.5 * (z1 - z2)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
