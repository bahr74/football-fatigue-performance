"""Small GPU convergence-and-cost pilot for GRU / LSTM (REFUSED without ``--approval``; nothing runs automatically).

Purpose: measure real GPU seconds per epoch, peak memory and convergence (best epoch under a 400-epoch cap with patience 25) for
the historical sizes, a parameter-matched LSTM and the largest search sizes, on one outer fold, seed 0, before any full benchmark
budget is approved. Every number it produces is labelled PILOT. Output: ``data/processed/m11-fair-gpu-pilot-v1/`` (write-once,
resumable: re-running the same command continues from saved per-epoch states).

Stop conditions (checked by the runner):
* NaN / inf training loss -> that configuration stops, recorded as failed;
* after 5 epochs, measured s/epoch > 3 x the estimate -> configuration skipped, recorded (no silent budget creep);
* peak GPU memory > ``MAX_GPU_MB`` -> stop the pilot;
* wall clock > ``MAX_WALL_H`` -> stop after the current configuration;
* another process using the GPU (Session B) -> refuse to start (``nvidia-smi`` check).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import distributions as DI
from . import halving as HV
from .nets import HISTORICAL

OUT = "m11-fair-gpu-pilot-v1"
FOLDS = ("0",)
SEEDS = (0,)
EPOCH_CAP, PATIENCE = 400, 25
MAX_GPU_MB, MAX_WALL_H, SLOW_FACTOR, PROBE_EPOCHS = 3500, 4.0, 3.0, 5
HIST_GPU_S_PER_EPOCH = {"gru": 1.24, "lstm": 3.15}          # measured on the GTX 1650 Ti, historical sizes, 22 fit matches
BASE = dict(lr=1e-3, weight_decay=1e-4, batch=256, dropout=0.1, emb=48, head=96, epochs=EPOCH_CAP, patience=PATIENCE)


def pilot_configs() -> list[dict[str, Any]]:
    from .fold_v2 import net_config
    g48 = HV.param_count(HISTORICAL["gru_gaplast"])
    h_pm, _ = HV.param_matched_hidden(g48, "lstm", lo=30, hi=60)
    rows = [("gru", 48, 1, "centred", "historical GRU size"), ("lstm", 40, 1, "centred", "historical LSTM size"),
            ("lstm", h_pm, 1, "centred", "LSTM parameter-matched to GRU-48"), ("gru", 96, 2, "centred", "largest GRU in the search"),
            ("lstm", 96, 2, "centred", "largest LSTM in the search"), ("gru", 48, 1, "shift_mix", "GRU-48 with the shift-mix distribution"),
            ("lstm", h_pm, 1, "shift_mix", "parameter-matched LSTM with the shift-mix distribution")]
    out = []
    for fam, h, L, dist, why in rows:
        c = dict(BASE, family=fam, hidden=h, layers=L, rnn_dropout=0.1 if L > 1 else 0.0, distribution=dist, why=why,
                 config_id=f"pilot-{fam}-h{h}-l{L}-{dist}")
        c["params"] = HV.param_count(net_config(fam, c))
        c["est_s_per_epoch"] = round(HIST_GPU_S_PER_EPOCH[fam] * max(1.0, c["params"] / (43585 if fam == "gru" else 41985)) ** 0.5, 2)
        out.append(c)
    return out


def estimate() -> dict[str, Any]:
    cfgs = pilot_configs()
    worst = sum(c["est_s_per_epoch"] * EPOCH_CAP for c in cfgs) * len(FOLDS) * len(SEEDS) / 3600
    likely = sum(c["est_s_per_epoch"] * min(EPOCH_CAP, 250) for c in cfgs) * len(FOLDS) * len(SEEDS) / 3600
    return dict(configs=cfgs, folds=list(FOLDS), seeds=list(SEEDS), epoch_cap=EPOCH_CAP, patience=PATIENCE,
                gpu_hours_worst_case=round(worst, 2), gpu_hours_if_best_epoch_about_225=round(likely, 2),
                estimate_basis="historical measured GPU s/epoch at the historical sizes, scaled by sqrt(parameter ratio) for larger "
                               "networks (an assumption the pilot replaces with measurements); CPU timing is NOT used (on CPU the GRU is "
                               "slower than the LSTM, the opposite of the GPU measurements)",
                expected_peak_gpu_mb="< 1500 (inputs on the GPU: 46 MB centred, 369 MB shift-mix; networks < 0.3 M parameters) [EST]",
                stop_conditions=dict(max_gpu_mb=MAX_GPU_MB, max_wall_h=MAX_WALL_H, slow_factor=SLOW_FACTOR, probe_epochs=PROBE_EPOCHS),
                outputs=f"data/processed/{OUT}/<distribution>/<config_id>/fold<k>.json, fold<k>_seed0.json (history, diagnostics, "
                        "test predictions on all stress tables), checkpoints/*.pt + sha256 sidecars, ledger.json",
                resume="re-run the same command: per-epoch training states are reused; finished folds are skipped")


def gpu_busy() -> list[str]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
        if r.returncode != 0:
            return [f"nvidia-smi failed (exit {r.returncode}); GPU state unknown"]
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception as e:                       # fail closed: an unknown GPU state blocks the pilot (M13 fix)
        return [f"nvidia-smi unavailable ({type(e).__name__}); GPU state unknown"]


def run(approval: str, device: str = "cuda", log=print) -> dict[str, Any]:
    import torch

    from football_fatigue.m11_phase2 import common as C

    from . import fold_v2 as F2
    from . import protocol as PR
    from .evaluation import EvalSets
    if not torch.cuda.is_available() and device == "cuda":
        raise SystemExit("CUDA is not available")
    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another process is using the GPU (Session B?): {busy}; not started")
    parts = PR.partition()
    if not PR.validate_partition(parts)["passed"]:
        raise SystemExit("partition validation failed")
    root = C.PROC / OUT
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "plan.json").exists():
        (root / "plan.json").write_text(json.dumps(dict(estimate(), approval=approval), indent=1), encoding="utf-8")
    evals = EvalSets(sequences=True, features=False)
    pools: dict[str, DI.Pool] = {}
    t0, ledger = time.perf_counter(), []
    for c in pilot_configs():
        if (time.perf_counter() - t0) / 3600 > MAX_WALL_H:
            ledger.append(dict(config_id=c["config_id"], status="not run: wall-clock limit"))
            continue
        dist = DI.DISTRIBUTIONS[c["distribution"]]
        if c["distribution"] not in pools:
            pools.clear()
            pools[c["distribution"]] = DI.load_pool(dist, sequences=True, features=False)
        pool = pools[c["distribution"]]
        out = root / c["distribution"] / c["config_id"]
        for f in FOLDS:
            # speed probe: PROBE_EPOCHS epochs into the resumable search state, then continue (no recomputation)
            probe = F2.neural(c["family"], c, 0, device)
            es = F2.primary_rows(pool, parts[f]["early_stop"])
            fit = pool.eligible(F2.primary_rows(pool, parts[f]["fit"]), dist)
            st = out / f"fold{f}_search" / f"{c['config_id']}.state.pt"
            st.parent.mkdir(parents=True, exist_ok=True)
            m = probe.fit(pool.si, pool.so, pool.gap, pool.y, pool.si[es], pool.so[es], pool.gap[es], pool.y[es],
                          draw=lambda rng: pool.draw(fit, dist, rng), state_path=str(st), stop_at_epoch=PROBE_EPOCHS)
            spe = m.diagnostics["s_per_epoch"]
            if spe > SLOW_FACTOR * c["est_s_per_epoch"]:
                ledger.append(dict(config_id=c["config_id"], fold=f, status="skipped: slow", s_per_epoch=spe, estimate=c["est_s_per_epoch"]))
                continue
            r = F2.run_fold_v2(c["family"], f, parts[f], pool, dist, evals, [c], SEEDS, out, search="full", device=device, log=log)
            d = r["per_seed"]["0"]["diagnostics"]
            ledger.append(dict(config_id=c["config_id"], fold=f, status="done", params=c["params"], s_per_epoch=d.get("s_per_epoch"),
                               est_s_per_epoch=c["est_s_per_epoch"], epochs_run=d.get("epochs_run"), best_epoch=d.get("best_epoch"),
                               at_cap=d.get("at_cap"), overfitting=d.get("overfitting"), peak_gpu_mb=d.get("peak_gpu_mb"),
                               gpu_seconds=r["runtime_s"]))
            if (d.get("peak_gpu_mb") or 0) > MAX_GPU_MB:
                ledger.append(dict(status=f"stopped: peak GPU memory {d['peak_gpu_mb']} MB > {MAX_GPU_MB}"))
                break
        (root / "ledger.json").write_text(json.dumps(ledger, indent=1, default=str), encoding="utf-8")
    return dict(ledger=ledger, wall_h=round((time.perf_counter() - t0) / 3600, 2), label="PILOT")
