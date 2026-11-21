"""Configurable pair networks (GRU / LSTM / MLP encoders, gap-last reading) for the fair benchmark.

At ``layers=1`` and the historical sizes, ``make_pair_net`` creates exactly the module tree and parameter order of
``m11_benchmark.models.make_net`` (so a seeded initialisation and the forward pass are identical; tested)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NetConfig:
    kind: str = "gru"            # gru | lstm | mlp
    hidden: int = 48
    emb: int = 48
    layers: int = 1
    head: int = 96
    dropout: float = 0.1         # head dropout (historical)
    rnn_dropout: float = 0.0     # between stacked recurrent layers (only when layers > 1)
    gap_last: bool = True
    T: int = 30

    def as_dict(self):
        return asdict(self)


HISTORICAL = {"gru_gaplast": NetConfig("gru", 48, 48), "lstm_gaplast": NetConfig("lstm", 40, 48)}


def make_pair_net(cfg: NetConfig):
    import torch
    nn = torch.nn
    kind, T, hidden, emb, head = cfg.kind, cfg.T, cfg.hidden, cfg.emb, cfg.head
    if kind not in ("gru", "lstm", "mlp"):
        raise ValueError(kind)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "mlp":
                self.f = nn.Sequential(nn.Flatten(), nn.Linear(T * 4, hidden), nn.ReLU(), nn.Linear(hidden, emb), nn.ReLU())
            else:
                kw = dict(num_layers=cfg.layers, dropout=cfg.rnn_dropout) if cfg.layers > 1 else {}
                self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(4, hidden, batch_first=True, **kw)
                self.proj = nn.Sequential(nn.Linear(hidden, emb), nn.ReLU())

        def forward(self, seq):
            if kind == "mlp":
                return self.f(seq)
            if cfg.gap_last:
                seq = seq.flip(1)
            out, _ = self.rnn(seq)
            m = seq[..., 2] > 0.5
            last = (m * torch.arange(seq.shape[1], device=seq.device)).argmax(1)
            return self.proj(out[torch.arange(len(seq), device=seq.device), last])

    class PairNet(nn.Module):
        """Per-link scores s[i, j] from (incoming i, outgoing j) embeddings; output = logit(SWAPPED) = s01 + s10 - s00 - s11.
        ``link_scores`` exposes the per-link matrix (not a calibrated per-link probability)."""

        def __init__(self):
            super().__init__()
            self.enc_in, self.enc_out = Enc(), Enc()
            self.head = nn.Sequential(nn.Linear(4 * emb + 1, head), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(head, head // 2), nn.ReLU(),
                                      nn.Linear(head // 2, 1))

        def link_scores(self, si, so, gap):
            B = si.shape[0]
            ei = self.enc_in(si.reshape(B * 2, *si.shape[2:])).reshape(B, 2, -1)
            eo = self.enc_out(so.reshape(B * 2, *so.shape[2:])).reshape(B, 2, -1)
            s = {}
            for i in (0, 1):
                for j in (0, 1):
                    a, b = ei[:, i], eo[:, j]
                    s[i, j] = self.head(torch.cat([a, b, a * b, (a - b).abs(), gap[:, None]], 1))[:, 0]
            return s

        def forward(self, si, so, gap):
            s = self.link_scores(si, so, gap)
            return s[0, 1] + s[1, 0] - s[0, 0] - s[1, 1]

    return PairNet()


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))
