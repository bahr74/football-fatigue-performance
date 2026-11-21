"""Models for Stage 4. Every model outputs p = P(SWAPPED) and is exactly symmetric under exchanging the two outgoing tracklets
(p -> 1 - p).

- Physics (B0-B3): ``motion`` cost margins m = cost(SWAPPED) - cost(MATCHED) (positive favours MATCHED); the raw decision is
  SWAPPED when m < 0; calibrated p = sigmoid(-w m / (0.25 + gap)) with one weight w >= 0 fitted on training data (no intercept,
  fitted on both outgoing orders, so the calibration keeps the antisymmetry).
- Tabular (LR, KNN, RBF-SVM via Nystroem, RF, HGB) on the motion features: trained on both outgoing orders (perm_* features
  negated, label flipped) and predicted symmetrically p = (f(x) + 1 - f(swap x)) / 2.
- Neural pair encoders (MLP / LSTM / GRU): tracklet encoders, a pair head s(i, j), logit(SWAPPED) = s01 + s10 - s00 - s11
  (exactly antisymmetric under either order exchange).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from football_fatigue.m11_recovery.motion import BASELINES

from . import data as D
from . import metrics as MT


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ---------------------------------------------------------------- physics

def physics_x(margins: np.ndarray, gap: np.ndarray, k: int) -> np.ndarray:
    return margins[:, k] / (0.25 + gap)


W_MIN = 1e-4


def fit_scale(x: np.ndarray, y: np.ndarray, rule: bool = True) -> float:
    """Weight w for p = sigmoid(-w x) by logistic regression without intercept on both orders (x, y) and (-x, 1 - y).

    ``rule=True`` (B0-B2, fixed physical rules): w >= W_MIN, so the decision is always the rule's own (lower cost assignment) and
    only the confidence is calibrated; a rule that is wrong more often than right keeps its decisions and gets near-chance
    probabilities. ``rule=False`` (B3): the sign is learned from the training data."""
    lr = LogisticRegression(C=1e3, fit_intercept=False, max_iter=1000).fit(np.r_[x, -x][:, None], np.r_[y, 1 - y])
    w = -float(lr.coef_[0, 0])
    return max(W_MIN, w) if rule else w


class PhysicsCombo:
    """B3 combination: logistic regression (no intercept, both orders) on the 8 normalised baseline margins."""

    def fit(self, M, gap, y):
        X = M / (0.25 + gap[:, None])
        self.sd = X.std(0) + 1e-9
        X = X / self.sd
        self.lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(np.r_[X, -X], np.r_[y, 1 - y])
        return self

    def predict(self, M, gap):
        X = M / (0.25 + gap[:, None]) / self.sd
        return 0.5 * (self.lr.predict_proba(X)[:, 1] + 1 - self.lr.predict_proba(-X)[:, 1])


def physics_predict(x: np.ndarray, w: float) -> np.ndarray:
    return _sigmoid(-w * x)


# ---------------------------------------------------------------- tabular

TABULAR_GRIDS: dict[str, list[dict[str, Any]]] = {
    "lr": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "knn": [dict(k=k) for k in (15, 31, 63)],
    "svm": [dict(gamma=g, C=c) for g in (0.5, 1.0, 2.0) for c in (0.1, 1.0)],
    "rf": [dict(max_depth=d) for d in (None, 12)],
    "hgb": [dict(learning_rate=0.05, max_leaf_nodes=31, max_iter=400), dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=400),
            dict(learning_rate=0.1, max_leaf_nodes=31, max_iter=200), dict(learning_rate=0.1, max_leaf_nodes=15, max_iter=200)],
}
STOCHASTIC = {"svm", "rf", "hgb"}


def make_tabular(name: str, cfg: dict[str, Any], n_features: int, seed: int):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=2000))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=cfg["k"]))
    if name == "svm":
        return make_pipeline(StandardScaler(), Nystroem(gamma=cfg["gamma"] / n_features, n_components=1000, random_state=seed),
                             LinearSVC(C=cfg["C"], dual=False, max_iter=5000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=cfg["max_depth"], min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
                                      random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=seed, **cfg)
    raise ValueError(name)


class Tabular:
    """Antisymmetric-augmented fit, symmetric prediction. SVM decision values are mapped to probabilities by a no-intercept
    sigmoid fitted on out-of-sample (inner-fold) decision values (``platt``)."""

    def __init__(self, name: str, cfg: dict[str, Any], k_perm: int, seed: int = 0):
        self.name, self.cfg, self.k, self.seed, self.platt = name, cfg, k_perm, seed, None

    def fit(self, X, y):
        X = np.nan_to_num(X)
        self.m = make_tabular(self.name, self.cfg, X.shape[1], self.seed).fit(np.r_[X, D.swap_features(X, self.k)], np.r_[y, 1 - y])
        return self

    def raw(self, X):
        """Symmetric score: probability (or SVM decision value) of SWAPPED, antisymmetrised."""
        X = np.nan_to_num(X)
        Z = D.swap_features(X, self.k)
        if self.name == "svm":
            return 0.5 * (self.m.decision_function(X) - self.m.decision_function(Z))
        return 0.5 * (self.m.predict_proba(X)[:, 1] + 1 - self.m.predict_proba(Z)[:, 1])

    def predict(self, X):
        r = self.raw(X)
        if self.name == "svm":
            if self.platt is None:
                raise RuntimeError("SVM needs a Platt weight from out-of-sample decision values")
            return _sigmoid(self.platt * r)
        return r


def fit_platt(d: np.ndarray, y: np.ndarray) -> float:
    lr = LogisticRegression(C=1e3, fit_intercept=False, max_iter=1000).fit(np.r_[d, -d][:, None], np.r_[y, 1 - y])
    return float(lr.coef_[0, 0])


# ---------------------------------------------------------------- neural pair encoders

NEURAL = {"mlp": dict(hidden=96, emb=48), "lstm": dict(hidden=40, emb=48), "gru": dict(hidden=48, emb=48),
          # corrective variants (same sizes and training): the recurrent encoder reads each tracklet from its far end towards the gap,
          # so the summary state is taken at the gap-adjacent valid point instead of the far end
          "gru_gaplast": dict(hidden=48, emb=48), "lstm_gaplast": dict(hidden=40, emb=48)}


def make_net(kind: str, T: int = 30, hidden: int = 48, emb: int = 48, head: int = 96, dropout: float = 0.1):
    import torch
    nn = torch.nn
    gap_last = kind.endswith("_gaplast")
    kind = kind.split("_")[0]

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "mlp":
                self.f = nn.Sequential(nn.Flatten(), nn.Linear(T * 4, hidden), nn.ReLU(), nn.Linear(hidden, emb), nn.ReLU())
            else:
                self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(4, hidden, batch_first=True)
                self.proj = nn.Sequential(nn.Linear(hidden, emb), nn.ReLU())

        def forward(self, seq):                               # (N, T, 4) -> (N, emb)
            if kind == "mlp":
                return self.f(seq)
            if gap_last:
                seq = seq.flip(1)                              # inputs start at the gap; reversed, the gap-adjacent point is last
            out, _ = self.rnn(seq)
            m = seq[..., 2] > 0.5                              # last valid step (masked steps never used as the summary)
            last = (m * torch.arange(seq.shape[1], device=seq.device)).argmax(1)
            return self.proj(out[torch.arange(len(seq), device=seq.device), last])

    class PairNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_in, self.enc_out = Enc(), Enc()
            self.head = nn.Sequential(nn.Linear(4 * emb + 1, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, head // 2), nn.ReLU(),
                                      nn.Linear(head // 2, 1))

        def forward(self, si, so, gap):                       # (B, 2, T, 4) x2, (B,) -> logit(SWAPPED)
            B = si.shape[0]
            ei = self.enc_in(si.reshape(B * 2, *si.shape[2:])).reshape(B, 2, -1)
            eo = self.enc_out(so.reshape(B * 2, *so.shape[2:])).reshape(B, 2, -1)
            s = {}
            for i in (0, 1):
                for j in (0, 1):
                    a, b = ei[:, i], eo[:, j]
                    s[i, j] = self.head(torch.cat([a, b, a * b, (a - b).abs(), gap[:, None]], 1))[:, 0]
            return s[0, 1] + s[1, 0] - s[0, 0] - s[1, 1]

    return PairNet()


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


class Neural:
    def __init__(self, kind: str, seed: int = 0, epochs: int = 80, patience: int = 10, batch: int = 256, lr: float = 1e-3,
                 weight_decay: float = 1e-4, clip: float = 1.0, dropout: float = 0.1):
        self.kind, self.seed, self.epochs, self.patience, self.batch, self.lr = kind, seed, epochs, patience, batch, lr
        self.wd, self.clip, self.dropout = weight_decay, clip, dropout

    def _dev(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, SI, SO, G, y, SIv, SOv, Gv, yv, weights=None, log=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self._dev()
        self.net = make_net(self.kind, dropout=self.dropout, **NEURAL[self.kind]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
        si, so, g, yy = f(SI), f(SO), f(G.astype(np.float32)), f(y.astype(np.float32))
        w = f((np.ones(len(y)) if weights is None else weights).astype(np.float32))
        best, best_state, bad, hist = np.inf, None, 0, []
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(y))
            tot = 0.0
            for s0 in range(0, len(perm), self.batch):
                idx = f(perm[s0:s0 + self.batch])
                a, b, t, gg, ww = si[idx].clone(), so[idx].clone(), yy[idx], g[idx], w[idx]
                if rng.random() < 0.5:                                  # reflection y -> -y (physics invariant)
                    a[..., 1] *= -1
                    b[..., 1] *= -1
                if rng.random() < 0.5:                                  # exchange outgoing order: label flips
                    b, t = b.flip(1), 1 - t
                if rng.random() < 0.5:                                  # exchange incoming order: label flips
                    a, t = a.flip(1), 1 - t
                opt.zero_grad()
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(self.net(a, b, gg), t, reduction="none") * ww).sum() / ww.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            pv = self.predict(SIv, SOv, Gv)
            vll = float(MT.ll_vec(yv, pv).mean())
            hist.append(dict(epoch=ep + 1, train_loss=tot / len(y), val_log_loss=vll))
            if vll < best - 1e-4:
                best, bad = vll, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.history, self.best_epoch = hist, int(np.argmin([h["val_log_loss"] for h in hist]) + 1)
        return self

    def predict(self, SI, SO, G):
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for s0 in range(0, len(G), 4096):
                f = lambda a: torch.as_tensor(np.ascontiguousarray(a[s0:s0 + 4096]), device=dev)  # noqa: E731
                out.append(torch.sigmoid(self.net(f(SI), f(SO), f(G.astype(np.float32)))).cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
