"""``python -m football_fatigue.m11_explorer <command>``

* ``nxm-benchmark``  synthetic curriculum benchmark of the deterministic Kalman + Hungarian(-null) solver -> m11-nxm-synthetic-benchmark-v1
* ``nxm-examples``   write synthetic example scenarios to docs/research/m11_multiplayer_examples/ (refuses existing files)
* ``selfcross``      self-crossing catalogue of the 32 development matches -> m11-self-crossings-v1 (CPU, ~30-40 min)
* ``nxm-dataset``    EXPLORATORY provider-derived N x M scenarios from real groups -> m11-nxm-exploratory-v1 (CPU, ~10 min)
* ``nxm-compare``    N x M solver variants + learned link scorers on the exploratory prototype -> m11-nxm-solver-comparison-v1
* ``review-plan``    prioritised human-review queue PROPOSAL -> m11-review-priority-v1 (nothing is marked reviewed)
* ``nxm-dataset-v2``  corrected EXPLORATORY N x M dataset (canonical encounters, repeat flag, decoupled timing) -> m11-nxm-exploratory-v2
* ``nxm-assoc-v2``    N x M benchmark with fit / calibration / test partitions on v1 and v2 -> m11-nxm-association-v2
* ``review-export``   review queue v2 (tiers 1-5 unchanged, tier 6 from v2) + CSV / JSON export -> m11-review-priority-v2
* ``window``         print every event in one window: ``window --match 3812 --period 1 --start 744 --length 10 --radius 0.5``
"""

from __future__ import annotations

import argparse
import json
import time

from football_fatigue.m11_phase2 import common as C


def nxm_benchmark(n: int) -> None:
    from . import multiplayer as MP
    out = C.Out("m11-nxm-synthetic-benchmark-v1")
    t0 = time.perf_counter()
    lv = MP.synthetic_benchmark(n)
    res = dict(schema="m11-nxm-synthetic-benchmark-v1", n_per_level=n, levels=lv, solver="Kalman forward (sigma_a 8, sigma_m 0.05) + Hungarian, "
               "Mahalanobis d2 cost, chi2(2, 0.99) gate, null option at the gate cost; forced = no null option",
               note=("SYNTHETIC motion (smooth Ornstein-Uhlenbeck turning) is easier than football: level 1 is not comparable with the real "
                     "0.5 m / 0.8 s population (Kalman forward 0.667 BA). These numbers test the solver and metrics, not real performance."),
               runtime_s=round(time.perf_counter() - t0, 1))
    out.json("results.json", res)
    out.manifest(parameters=dict(n_per_level=n, levels=list(MP.LEVELS)), code=_code())
    print(json.dumps({k: v["kalman_hungarian_null"] for k, v in lv.items()}, indent=1))


def nxm_examples() -> None:
    from . import multiplayer as MP
    d = C.ROOT / "docs" / "research" / "m11_multiplayer_examples"
    d.mkdir(parents=True, exist_ok=True)
    for lv, seed in ((1, 0), (3, 2), (5, 1), (7, 4)):
        p = d / f"synthetic_L{lv}_seed{seed}.json"
        if p.exists():
            raise SystemExit(f"{p} exists; not overwritten")
        p.write_text(json.dumps(MP.generate(lv, seed), indent=1), encoding="utf-8")
        print(p)


def selfcross() -> None:
    import pandas as pd
    from . import selfcross as SX
    from . import tracks as TR
    out = C.Out("m11-self-crossings-v1")
    cfg = SX.SelfCrossConfig()
    parts, per = [], {}
    t0 = time.perf_counter()
    for m in C.allowed():
        t1 = time.perf_counter()
        M = TR.load(m)
        frames = []
        for p in M.periods()["period"]:
            df = M.self_crossings(int(p), -1e12, 1e12, cfg=cfg)                    # whole period, every chain
            df.insert(0, "match_id", int(m))
            frames.append(df)
        d = pd.concat(frames, ignore_index=True)
        parts.append(d)
        per[m] = dict(n=len(d), low_quality=int((d["quality"] == "low").sum()), seconds=round(time.perf_counter() - t1, 1))
        print(m, per[m], flush=True)
        TR.load.cache_clear()
    allp = pd.concat(parts, ignore_index=True)
    out.parquet("self_crossings.parquet", allp)
    out.json("summary.json", dict(schema="m11-self-crossings-v1", config=cfg.as_dict(), per_match=per, total=len(allp),
                                  dt_s=C.q(allp["dt_s"]), path_m=C.q(allp["path_m"]), extent_m=C.q(allp["extent_m"]),
                                  note="descriptive one-player loops; never two-player events, association candidates or occlusion",
                                  runtime_s=round(time.perf_counter() - t0, 1)))
    out.manifest(inputs=C.catalogue_inputs(C.allowed()), parameters=cfg.as_dict(), code=_code())


def window(a) -> None:
    from . import events as EV
    W = EV.window_events(a.match, a.period, a.start, a.start + a.length, a.radius)
    print(json.dumps(W.counts(), indent=1))
    print(W.edges[["edge_id", "slot_a", "slot_b", "start", "end", "d_min", "context"]].to_string(index=False))
    print(W.crossings[["crossing_id", "event_type", "slot_a", "slot_b", "ta", "tb"]].to_string(index=False))


def _code():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    return {f"m11_explorer/{p.name}": C.sha256_file(p) for p in sorted(here.glob("*.py"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_explorer")
    ap.add_argument("command", choices=["nxm-benchmark", "nxm-examples", "selfcross", "window", "nxm-dataset", "nxm-compare", "review-plan",
                                        "nxm-dataset-v2", "nxm-assoc-v2", "review-export"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--match")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--start", type=float)
    ap.add_argument("--length", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.command == "nxm-dataset":
        from . import nxm_dataset as ND
        return ND.main()
    if a.command == "review-plan":
        from . import review_plan as RP
        return RP.main()
    if a.command == "nxm-dataset-v2":
        from . import nxm_dataset_v2 as N2
        return N2.main()
    if a.command == "nxm-assoc-v2":
        from . import nxm_assoc_v2 as A2
        return A2.main()
    if a.command == "review-export":
        from . import review_v2 as RV
        return RV.main()
    if a.command == "nxm-compare":
        from . import nxm_compare as NC
        return NC.main()
    {"nxm-benchmark": lambda: nxm_benchmark(a.n), "nxm-examples": nxm_examples, "selfcross": selfcross, "window": lambda: window(a)}[a.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
