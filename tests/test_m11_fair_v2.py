"""Fair benchmark v2 infrastructure (synthetic, CPU): historical equivalence, resume, checkpoints, convergence diagnostics,
successive halving, capacity / compute matching, training distributions and the fold runner."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from football_fatigue.m11_fair import distributions as DI  # noqa: E402
from football_fatigue.m11_fair import halving as HV  # noqa: E402
from football_fatigue.m11_fair import train as TR  # noqa: E402
from football_fatigue.m11_fair.nets import HISTORICAL, NetConfig, make_pair_net  # noqa: E402


def _seq(n=240, T=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 15
    p0, v = rng.normal(0, 1, (n, 2, 2)), rng.normal(0, 3, (n, 2, 2))
    si = np.zeros((n, 2, T, 4), np.float32)
    so = np.zeros((n, 2, T, 4), np.float32)
    si[..., :2] = (p0[:, :, None, :] - v[:, :, None, :] * t[None, None, :, None]) / 10
    so[..., :2] = (p0[:, :, None, :] + v[:, :, None, :] * (0.8 + t[None, None, :, None])) / 10
    si[..., 2] = so[..., 2] = 1
    si[..., 3], so[..., 3] = -t, t
    y = rng.integers(0, 2, n)
    so[y == 1] = so[y == 1][:, ::-1]
    return si, so, np.full(n, 0.8, np.float32), y


def _fit(model, si, so, g, y, **kw):
    return model.fit(si[:160], so[:160], g[:160], y[:160], si[160:], so[160:], g[160:], y[160:], **kw)


def test_default_training_path_equals_the_historical_trainer(monkeypatch):
    from football_fatigue.m11_benchmark import models as MD
    monkeypatch.setattr(MD.Neural, "_dev", lambda self: "cpu")
    si, so, g, y = _seq()
    a = MD.Neural("gru_gaplast", seed=3, epochs=3, patience=10)
    a.fit(si[:160], so[:160], g[:160], y[:160], si[160:], so[160:], g[160:], y[160:])
    b = _fit(TR.FairNeural(HISTORICAL["gru_gaplast"], epochs=3, patience=10, seed=3, device="cpu"), si, so, g, y)
    for k, v in a.net.state_dict().items():
        assert torch.allclose(v, b.net.state_dict()[k], atol=1e-6), k


def test_resume_is_bit_identical(tmp_path):
    si, so, g, y = _seq()
    cfg = NetConfig("gru", 16, 16)
    full = _fit(TR.FairNeural(cfg, epochs=6, patience=50, seed=1, device="cpu"), si, so, g, y, perturb=((0.0, 0.05), (0.0, 0.2)),
                draw=lambda rng: rng.choice(160, 160))
    st = tmp_path / "s.pt"
    part = _fit(TR.FairNeural(cfg, epochs=6, patience=50, seed=1, device="cpu"), si, so, g, y, perturb=((0.0, 0.05), (0.0, 0.2)),
                draw=lambda rng: rng.choice(160, 160), state_path=str(st), stop_at_epoch=3)
    assert not part.finished and len(part.history) == 3
    rest = _fit(TR.FairNeural(cfg, epochs=6, patience=50, seed=1, device="cpu"), si, so, g, y, perturb=((0.0, 0.05), (0.0, 0.2)),
                draw=lambda rng: rng.choice(160, 160), state_path=str(st))
    assert rest.finished and [h["es_log_loss"] for h in rest.history] == [h["es_log_loss"] for h in full.history]
    for k, v in full.net.state_dict().items():
        assert torch.equal(v, rest.net.state_dict()[k]), k


def test_checkpoint_integrity_and_no_overwrite(tmp_path):
    cfg = NetConfig("lstm", 12, 8)
    net = make_pair_net(cfg)
    p = tmp_path / "c" / "m.pt"
    side = TR.save_checkpoint(net, p, dict(family="lstm"))
    assert side["params"] == sum(x.numel() for x in net.parameters()) and TR.verify_checkpoint(p, cfg)
    with pytest.raises(FileExistsError):
        TR.save_checkpoint(net, p, {})
    b = bytearray(p.read_bytes())
    b[-5] ^= 0xFF
    p.write_bytes(bytes(b))
    assert not TR.verify_checkpoint(p, cfg)


def test_convergence_diagnostics_flag_cap_overfitting_and_patience():
    mk = lambda es, tr: [dict(epoch=i + 1, train_loss=t, es_log_loss=e) for i, (e, t) in enumerate(zip(es, tr))]  # noqa: E731
    at_cap = TR.convergence(mk(np.linspace(0.6, 0.3, 40), np.linspace(0.6, 0.2, 40)), 40, 10)
    assert at_cap["at_cap"] and at_cap["still_improving"] and not at_cap["overfitting"]
    es = np.r_[np.linspace(0.6, 0.3, 20), np.linspace(0.3, 0.36, 15)]
    over = TR.convergence(mk(es, np.linspace(0.6, 0.1, 35)), 100, 15)
    assert over["overfitting"] and over["stopped_by_patience"] and over["best_epoch"] == 20 and over["es_rise_after_best"] > 0.05


def test_successive_halving_schedule_budget_and_selection(tmp_path):
    assert HV.schedule(12) == [12, 4, 2, 1] and HV.max_epochs(12) == 12 * 20 + 4 * 40 + 2 * 120 + 1 * 220
    assert HV.max_epochs(24) < 24 * 400 / 2.5
    assert HV.configs_for_budget(HV.max_epochs(12) * 1.0, 1.0) == 12 and HV.configs_for_budget(HV.max_epochs(12) * 1.0, 3.0) < 12

    class Fake:
        def __init__(self, q, e):
            self.best_es_log_loss, self.finished, self.history, self.diagnostics = q, e >= 400, [{}] * e, dict(params=1)
    quality = {f"c{i}": 0.3 + 0.01 * i for i in range(9)}
    calls = []

    def train(cfg, stop, state):
        calls.append((cfg["config_id"], stop))
        return Fake(quality[cfg["config_id"]] - 0.0001 * stop, stop)
    r = HV.successive_halving([dict(config_id=k) for k in quality], train, tmp_path, eta=3, log=lambda s: None)
    assert r["selected"] == "c0" and [len(x["alive"]) for x in r["rungs"]] == [9, 3, 1, 1]
    assert sum(1 for c, s in calls if s == 400) == 1 and r["total_epochs"] == 6 * 20 + 2 * 60 + 400


def test_parameter_matched_lstm_is_close_to_the_gru():
    target = HV.param_count(HISTORICAL["gru_gaplast"])
    h, p = HV.param_matched_hidden(target, "lstm", lo=30, hi=50)
    assert abs(p - target) / target < 0.02 and h < 48


def test_pool_draws_one_row_per_encounter_from_allowed_variants():
    n0 = 50
    owner = np.r_[np.arange(n0), np.arange(n0)]
    tabs = np.r_[np.full(n0, "shiftaug", object), np.full(n0, "g0p40", object)]
    pool = DI.Pool(y=np.zeros(3 * n0, int), groups=np.repeat(["1"], 3 * n0), gap=np.zeros(3 * n0, np.float32), n0=n0, owner=owner,
                   variant_table=tabs)
    pool.index()
    rng = np.random.default_rng(0)
    dist = DI.TrainDist("t", ("shiftaug",), 0.5)
    d = pool.draw(np.arange(n0), dist, rng)
    assert len(d) == n0
    var = d[d >= n0] - n0
    assert len(var) > 10 and (tabs[var] == "shiftaug").all() and (owner[var] == np.flatnonzero(d >= n0)).all()
    assert (pool.draw(np.arange(n0), DI.DISTRIBUTIONS["centred"], rng) == np.arange(n0)).all()


def test_online_perturbation_keeps_gap_point_and_masked_steps():
    si, so, g, y = _seq(8)
    si[:, :, 25:, :] = 0                                                       # masked tail
    a, b = DI.perturb_batch(torch.as_tensor(si), torch.as_tensor(so), 0.05, 0.5, torch.Generator().manual_seed(0))
    assert (a[..., 0, 2] == 1).all() and (b[..., 0, 2] == 1).all()
    assert (a[:, :, 25:] == 0).all()
    assert 0.3 < 1 - float(a[:, :, 1:25, 2].mean()) < 0.7


class _Evals:
    def __init__(self, si, so, g, y, groups, F):
        self.sets = {t: dict(si=si, so=so, gap=g, y=y, groups=groups, F=F, candidate_id=np.array([f"c{i}" for i in range(len(y))]))
                     for t in ("g0p80", "noise5cm")}

    def rows(self, t, matches):
        return np.flatnonzero(np.isin(self.sets[t]["groups"], list(matches)))


@pytest.mark.parametrize("family", ["gru", "hgb"])
def test_fold_runner_v2_end_to_end_is_resumable(tmp_path, family):
    from football_fatigue.m11_fair import fold_v2 as F2
    from football_fatigue.m11_fair import spaces as SP
    si, so, g, y = _seq(360)
    groups = np.repeat([str(k) for k in range(12)], 30)
    F = np.c_[np.where(y == 1, -1, 1) * np.abs(np.random.default_rng(0).normal(1, 0.5, 360)), np.random.default_rng(1).normal(0, 1, 360)]
    pool = DI.Pool(y=y, groups=groups, gap=g, n0=360, si=si, so=so, F=F, k_perm=1, meta=None)
    pool.index()
    part = dict(fit=[str(k) for k in range(6)], early_stop=["6", "7"], calibration=["8", "9"], test=["10", "11"])
    cfgs = SP.sample(family, 3, salt="t")
    for c in cfgs:
        c.update(epochs=6, patience=3, hidden=12, layers=1, batch=64) if family == "gru" else c.update(max_iter=30)
    ev = _Evals(si, so, g, y, groups, F)
    r = F2.run_fold_v2(family, "0", part, pool, DI.DISTRIBUTIONS["centred"], ev, cfgs, (0, 1), tmp_path, rungs=(2, 4), eta=3, device="cpu",
                       log=lambda s: None)
    assert set(r["per_seed"]) == {"0", "1"} and r["roles"]["test"] == ["10", "11"]
    seed0 = json.loads((tmp_path / "fold0_seed0.json").read_text())
    assert len(seed0["predictions"]["g0p80"]["p"]) == 60 and set(seed0["predictions"]) == {"g0p80", "noise5cm"}
    if family == "gru":
        assert r["search_result"]["rungs"][0]["epoch"] == 2 and TR.verify_checkpoint(tmp_path / "checkpoints" / "gru_fold0_seed1.pt",
                                                                                       F2.net_config("gru", r["selected"]))
    again = F2.run_fold_v2(family, "0", part, pool, DI.DISTRIBUTIONS["centred"], ev, cfgs, (0, 1), tmp_path, device="cpu", log=lambda s: None)
    assert again == json.loads((tmp_path / "fold0.json").read_text())
