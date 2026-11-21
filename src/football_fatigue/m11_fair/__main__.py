"""``python -m football_fatigue.m11_fair <command>``

* ``plan``      write the fair-benchmark approval package (partition re-validated, 24 configurations per family, compute) [CPU, seconds]
* ``smoke``     end-to-end synthetic CPU run of every family (tiny data, 2 configurations, 2 epochs; writes nothing) [CPU, ~1 min]
* ``cpu-pilot`` reduced-budget tabular pilot (HGB, linear / Nystroem / exact-RBF SVM) on the 32 development matches [CPU, ~1 h]
* ``pilot-plan`` print / write the GPU pilot package (configurations, estimate, stop conditions; nothing is trained) [CPU, seconds]
* ``gpu-pilot`` the small GRU / LSTM convergence-and-cost pilot. REFUSED unless ``--approval <id>`` is given.
* ``train``     the fair benchmark v2 for one family (``--dist``, ``--search halving|full``, ``--configs``, ``--folds``, ``--seeds``).
                REFUSED unless ``--approval <id>`` is given (GPU training needs explicit approval).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def smoke() -> dict:
    """Synthetic rows: outgoing tracklets continue the incoming ones (y = 0) or are exchanged (y = 1)."""
    from . import runner as RU
    from . import spaces as SP
    rng = np.random.default_rng(0)
    n, T = 480, 30
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
    F = np.c_[np.where(y == 1, -1, 1)[:, None] * rng.uniform(0.2, 2, (n, 2)) + rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 2))]
    groups = np.repeat([str(k) for k in range(12)], n // 12)
    data = RU.DevData(y, groups, si, so, np.full(n, 0.8, np.float32), F, 2)
    part = dict(fit=[str(k) for k in range(6)], early_stop=["6", "7"], calibration=["8", "9"], test=["10", "11"])
    out = {}
    for fam in ("gru", "lstm", "hgb", "svm_linear", "svm_nystroem", "svm_rbf_exact"):
        cfgs = SP.sample(fam, 2, salt="smoke")
        for c in cfgs:
            if fam in RU.NEURAL:
                c.update(epochs=2, patience=2, hidden=16, layers=1)
            if fam == "hgb":
                c.update(max_iter=50)
            if fam == "svm_nystroem":
                c.update(n_components=100)
        r = RU.run_fold(fam, "0", part, data, cfgs, seeds=(0,), device="cpu", log=lambda s: None)
        p = np.asarray(r["per_seed"]["0"]["p_test"])
        out[fam] = dict(selected=r["selected"]["config_id"], n_test=len(p), finite=bool(np.isfinite(p).all()),
                        test_acc=round(float(((p >= 0.5) == y[np.asarray(r["test_rows"])]).mean()), 3))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_fair")
    ap.add_argument("command", choices=["plan", "smoke", "cpu-pilot", "pilot-plan", "gpu-pilot", "train"])
    ap.add_argument("--family", choices=["gru", "lstm", "hgb", "svm_nystroem", "svm_rbf_exact", "svm_linear"])
    ap.add_argument("--approval", default=None, help="approval id recorded with the run (required for train)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--configs", type=int, default=None, help="limit configurations (default: the full 24)")
    ap.add_argument("--dist", default="centred", help="training distribution (m11_fair.distributions.DISTRIBUTIONS)")
    ap.add_argument("--search", choices=["halving", "full"], default="halving")
    ap.add_argument("--folds", default=None, help="comma-separated outer folds (default: all 8)")
    ap.add_argument("--seeds", type=int, default=None, help="number of final seeds (default: 5 neural, 3 tabular)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    if a.command == "plan":
        from . import protocol as PR
        p = PR.main()
        print(json.dumps(dict(partition=p["partition_validation"], compute=p["compute_estimate_gpu_h"]), indent=1))
    elif a.command == "smoke":
        print(json.dumps(smoke(), indent=1))
    elif a.command == "cpu-pilot":
        from . import pilot as PI
        PI.main(a.workers)
    elif a.command == "pilot-plan":
        from . import gpu_pilot as GP
        from football_fatigue.m11_phase2 import common as C
        est = GP.estimate()
        p = C.PROC / "m11-fair-gpu-pilot-plan-v1" / "pilot_plan.json"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(est, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in est.items() if k != "configs"}, indent=1))
        for c in est["configs"]:
            print(c["config_id"], c["params"], "params, est", c["est_s_per_epoch"], "s/epoch")
    elif not a.approval:
        print("REFUSED: GPU training needs explicit approval. Re-run with --approval <id> once the protocol, the GPU budget and the "
              "family are approved (see docs/research/m11_fair_four_model_protocol_v1.md and "
              "docs/research/m12_gpu_pilot_approval_package_v1.md).", file=sys.stderr)
        return 2
    elif a.command == "gpu-pilot":
        from . import gpu_pilot as GP
        print(json.dumps(GP.run(a.approval, a.device or "cuda"), indent=1, default=str))
    else:
        if not a.family:
            ap.error("--family is required for train")
        from . import distributions as DI
        from . import fold_v2 as F2
        from . import protocol as PR
        from . import spaces as SP
        from .evaluation import EvalSets
        from football_fatigue.m11_phase2 import common as C
        parts = PR.partition()
        if not PR.validate_partition(parts)["passed"]:
            raise SystemExit("partition validation failed")
        dist = DI.DISTRIBUTIONS[a.dist]
        neural = a.family in F2.NEURAL
        pool = DI.load_pool(dist, sequences=neural, features=not neural)
        evals = EvalSets(sequences=neural, features=not neural)
        cfgs = SP.sample(a.family)[: a.configs or SP.N_CONFIGS]
        seeds = (PR.NEURAL_SEEDS if neural else PR.TABULAR_SEEDS)[: a.seeds or None]
        root = C.PROC / "m11-fair-benchmark-v2" / a.family / a.dist / f"{a.search}_{len(cfgs)}cfg"
        folds = a.folds.split(",") if a.folds else sorted(parts, key=int)
        for f in folds:
            r = F2.run_fold_v2(a.family, f, parts[f], pool, dist, evals, cfgs, seeds, root, search=a.search, device=a.device)
            print(f"fold {f}: {r['runtime_s']} s, selected {r['selected']['config_id']}")
        (root / "approval.txt").write_text(a.approval, encoding="utf-8") if not (root / "approval.txt").exists() else None
    print(f"finished in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
