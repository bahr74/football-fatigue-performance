"""Compute-efficient neural search: successive halving with resumable training, parameter matching and compute matching.

Successive halving (SH): all n configurations train to rung 1 (epochs ``rungs[0]``), the best ceil(n / eta) by best-so-far
early-stop log loss continue (resumed from their saved state, no recomputation) to the next rung, until the last rung (the epoch
cap; patience still ends training earlier). A configuration that stops by patience before a rung is finished and keeps its score.
Every epoch is recorded in a compute ledger (seconds, epochs) so GRU and LSTM costs are measured, not assumed.

Two comparisons are reported separately (never conflated):

* **equal configurations**: both families get the same configuration list size and the same SH schedule (equal *selection*
  opportunity; unequal compute because an LSTM epoch costs more);
* **equal compute**: both families get the same GPU-second budget; the number of configurations each can afford under the same
  schedule follows from its *measured* seconds per epoch (``configs_for_budget``).

Parameter matching: ``param_matched_hidden`` finds the hidden size of one family with the parameter count closest to a reference
network (the LSTM has 4/3 of the GRU's recurrent weights per hidden unit).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .nets import NetConfig, make_pair_net, n_params

DEFAULT_RUNGS = (20, 60, 180, 400)
ETA = 3


def schedule(n: int, rungs=DEFAULT_RUNGS, eta: int = ETA) -> list[int]:
    """Configurations alive at each rung."""
    alive, out = n, []
    for _ in rungs:
        out.append(alive)
        alive = max(1, math.ceil(alive / eta))
    return out


def max_epochs(n: int, rungs=DEFAULT_RUNGS, eta: int = ETA) -> int:
    """Upper bound on epochs trained (no patience stops): sum over rungs of alive x (rung - previous rung)."""
    prev, tot = 0, 0
    for a, r in zip(schedule(n, rungs, eta), rungs):
        tot += a * (r - prev)
        prev = r
    return tot


def configs_for_budget(budget_s: float, s_per_epoch: float, rungs=DEFAULT_RUNGS, eta: int = ETA, n_max: int = 64) -> int:
    """Largest n whose worst-case SH cost fits ``budget_s`` at the measured ``s_per_epoch`` (compute-matched comparison)."""
    n = 0
    for k in range(1, n_max + 1):
        if max_epochs(k, rungs, eta) * s_per_epoch <= budget_s:
            n = k
    return n


def param_count(cfg: NetConfig) -> int:
    return n_params(make_pair_net(cfg))


def param_matched_hidden(target: int, kind: str, layers: int = 1, emb: int = 48, head: int = 96, lo: int = 8, hi: int = 256) -> tuple[int, int]:
    best = None
    for h in range(lo, hi + 1):
        p = param_count(NetConfig(kind, h, emb, layers, head))
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (h, p)
    return best


def successive_halving(configs: list[dict[str, Any]], train: Callable[[dict[str, Any], int, Path], Any], state_dir: Path,
                       rungs=DEFAULT_RUNGS, eta: int = ETA, log: Callable[[str], None] = print) -> dict[str, Any]:
    """``train(cfg, stop_at_epoch, state_path)`` -> a fitted FairNeural paused at ``stop_at_epoch`` (or finished). Returns the
    chosen configuration id, per-rung survivors and the compute ledger."""
    state_dir.mkdir(parents=True, exist_ok=True)
    alive = [c["config_id"] for c in configs]
    by_id = {c["config_id"]: c for c in configs}
    score: dict[str, float] = {}
    finished: dict[str, bool] = {}
    ledger: dict[str, dict[str, Any]] = {}
    rung_log = []
    for ri, r in enumerate(rungs):
        for cid in alive:
            if finished.get(cid):
                continue
            t1 = time.perf_counter()
            m = train(by_id[cid], r, state_dir / f"{cid}.state.pt")
            score[cid], finished[cid] = m.best_es_log_loss, m.finished
            e = ledger.setdefault(cid, dict(epochs=0, seconds=0.0, params=m.diagnostics.get("params")))
            e["epochs"] = len(m.history)
            e["seconds"] = round(e["seconds"] + time.perf_counter() - t1, 2)
            e["diagnostics"] = m.diagnostics
            log(f"rung {ri} ({r} ep) {cid}: es_ll {score[cid]:.4f} epochs {e['epochs']} finished {finished[cid]}")
        ranked = sorted(alive, key=lambda c: score[c])
        rung_log.append(dict(rung=ri, epoch=r, alive=list(alive), scores={c: round(score[c], 5) for c in alive}))
        if ri < len(rungs) - 1:
            alive = ranked[:max(1, math.ceil(len(alive) / eta))]
    best = min(alive, key=lambda c: score[c])
    return dict(selected=best, rungs=rung_log, ledger=ledger, total_epochs=int(sum(v["epochs"] for v in ledger.values())),
                total_seconds=round(sum(v["seconds"] for v in ledger.values()), 1), schedule=list(rungs), eta=eta)
