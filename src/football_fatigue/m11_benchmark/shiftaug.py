"""Stage 4C shift-augmentation dataset (``data/processed/m11-stage4c-shiftaug-v1``; train/CV matches only, written once).

For every candidate of the primary Stage 3 table (``g0p80``) and every gap-centre shift in ``SHIFTS`` (+-0.1, +-0.2, +-0.3 s) the
Stage 3 procedure for shifted gaps is re-run unchanged (``samples`` functions, not copies):

1. the full Stage 2 label-quality procedure at the shifted centre (``candidates.label_quality``); only ACCEPTED shifts are built,
   every rejection is recorded with its reasons (continuous reference chains, movement plausibility, ...);
2. construction from the two players' own chain vertices (``samples.build_one``: no interpolation across the hidden gap, no
   bridging between chains, incoming-only frame);
3. independent label verification against the raw provider positions (``samples.verify_label``), gap-boundary and inverse-transform
   checks, and the hidden-gap perturbation test on a seeded subset.

Order bits are drawn per shift (``order_bits(candidate_id, "g0p80_aug<shift>")``), so neither order nor row position identifies a
shift. Each sample keeps its candidate id, match and the ``g0p80`` overlap group / independence flag: shifted versions are
augmentations of the same encounter, never additional independent observations. A reproduction check rebuilds, for each candidate
whose Stage 3 ``g0p80_shift`` shift is non-zero, that exact sample with the Stage 3 order bits and requires byte-identical arrays.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.encounters import quality as Q
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import core as W
from football_fatigue.window_intersections import local as L
from football_fatigue.window_intersections import samples as S

OUT = "m11-stage4c-shiftaug-v1"
SHIFTS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
GAP = 0.8
PERTURB_PER_SHIFT = 10


def skey(s: float) -> str:
    return f"{s:+.1f}"


def table_name(s: float) -> str:
    return f"g0p80_aug{skey(s)}"


def out_dir() -> Path:
    from football_fatigue.window_intersections.__main__ import PROC
    return PROC / OUT


def build_match_shifts(c, mask, dup, seg, ch, cand: pd.DataFrame, meta0: pd.DataFrame, s3_shift: dict[str, np.ndarray],
                       ccfg: C.CandidateConfig, seed: int) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cx, cy, _ = W.coefficients(seg, "linear")
    est = (mask & Q.ESTIMATED) != 0
    periods = {}
    for p in sorted(int(v) for v in np.unique(cand["period"])):
        rows_p = np.flatnonzero(c.canonical_period == p)
        tp = c.video_time_s[rows_p]
        periods[p] = (rows_p, tp, float(tp[0]), float(tp[-1]))
    m0 = meta0.set_index("candidate_id")
    s3_rows = {cid: i for i, cid in enumerate(s3_shift["candidate_id"])}
    rows, metas, fails = [], [], []
    checks: Counter = Counter()
    rng = np.random.default_rng(seed)
    cache: dict[int, tuple] = {}

    def verts(k):
        if k not in cache:
            cache[k] = S.chain_vertices(seg, ch, k)
        return cache[k]

    cand = cand[cand["candidate_id"].isin(m0.index)]
    for r in cand.itertuples(index=False):
        r = r._asdict()
        cid = r["candidate_id"]
        ka, kb = int(r["chain_a"]), int(r["chain_b"])
        if not (ch["slot"][ka] == r["slot_a"] and ch["slot"][kb] == r["slot_b"]):
            raise RuntimeError(f"{cid}: Stage 2 chain ids do not match slots")
        ta, xa, ya, rowa = verts(ka)
        tb, xb, yb, rowb = verts(kb)
        VA, VB = (ta, xa, ya), (tb, xb, yb)
        raw = {"A": (c.video_time_s[rowa], c.x[rowa, r["slot_a"]], c.y[rowa, r["slot_a"]]),
               "B": (c.video_time_s[rowb], c.x[rowb, r["slot_b"]], c.y[rowb, r["slot_b"]])}
        rows_p, tp, pstart, pend = periods[r["period"]]
        for sh in SHIFTS:
            q = C.label_quality(c, mask, dup, est, seg, ch, cx, cy, dict(r, t_star=r["t_star"] + sh), rows_p, tp, pstart, pend, GAP, ccfg)
            if q["outcome"] != C.ACCEPTED:
                why = json.loads(q["reasons"]) or json.loads(q["lower_reasons"])
                fails.append(dict(candidate_id=cid, shift_s=sh, stage="shift_revalidation", outcome=q["outcome"], reason="|".join(why)))
                continue
            tc = r["t_star"] + sh
            io_ = S.order_bits(cid, table_name(sh))
            s = S.build_one(VA, VB, tc, GAP, *io_)
            if "failure" in s:
                fails.append(dict(candidate_id=cid, shift_s=sh, stage="construction", outcome="", reason=s["failure"]))
                continue
            lab = S.verify_label(s, raw)
            checks[f"label_{lab}"] += 1
            gin_ok = bool((s["grid_in"][s["in_mask"].any(0)] < s["g0"]).all() and (s["grid_out"][s["out_mask"].any(0)] > s["g1"]).all())
            checks["gap_boundary_ok" if gin_ok else "gap_boundary_bad"] += 1
            inv = max(float(np.abs((s["in_pos"][i][s["in_mask"][i]].astype(np.float64) @ s["R"]) + s["origin"]
                                   - s["pitch_in"][i][s["in_mask"][i]]).max()) for i in (0, 1))
            checks["inverse_transform_ok" if inv < 1e-3 else "inverse_transform_bad"] += 1
            if lab != "ok" or not gin_ok or inv >= 1e-3:
                fails.append(dict(candidate_id=cid, shift_s=sh, stage="verification", outcome="", reason=f"label_{lab}"))
                continue
            if checks[f"perturbation_tested{skey(sh)}"] < PERTURB_PER_SHIFT and (rng.random() < 0.05 or checks[f"perturbation_tested{skey(sh)}"] < 2):
                checks[f"perturbation_tested{skey(sh)}"] += 1
                checks["perturbation_identical" if S.gap_perturbation_ok(VA, VB, s, tc, GAP, io_) else "perturbation_changed"] += 1
            if cid in s3_rows and abs(float(s3_shift["shift_s"][s3_rows[cid]]) - sh) < 1e-9:   # Stage 3 reproduction check
                s2 = S.build_one(VA, VB, tc, GAP, *S.order_bits(cid, "g0p80_shift"))
                j = s3_rows[cid]
                same = all(np.array_equal(S.model_arrays(s2, GAP)[k], s3_shift[k][j]) for k in S.ARRAY_FIELDS)
                checks["stage3_reproduction_identical" if same else "stage3_reproduction_different"] += 1
            rows.append(S.model_arrays(s, GAP))
            e0 = q[f"est_frac_in_gap_{'a' if s['who_in'][0] == 'A' else 'b'}"]
            e1 = q[f"est_frac_in_gap_{'b' if s['who_in'][0] == 'A' else 'a'}"]
            b = m0.loc[cid]
            metas.append(dict(candidate_id=cid, match_id=r["match_id"], shift_s=sh, gap_centre_s=tc, y=s["y"], in_order=io_[0], out_order=io_[1],
                              frame_mode=s["frame_mode"], overlap_group=int(b["overlap_group"]), independent=bool(b["independent"]),
                              y_centred=int(b["y"]), crossing_class=b["crossing_class"], d_min=float(b["d_min"]), d_band=b["d_band"],
                              speed_band=b["speed_band"], team_relation=b["team_relation"], status_in_gap_centred=b["status_in_gap"],
                              status_in_gap=("both_fully_estimated" if e0 == 1 and e1 == 1 else "both_fully_visible" if e0 == 0 and e1 == 0
                                             else "mixed" if np.isfinite(e0) and np.isfinite(e1) else "unknown"),
                              est_frac_in_gap_in0=e0, est_frac_in_gap_in1=e1, n_valid_min=int(min(s["n_valid"]))))
    A = S.stack(rows)
    M = pd.DataFrame(metas)
    if len(M):
        M.insert(0, "row", np.arange(len(M)))
    F = pd.DataFrame(fails, columns=["candidate_id", "shift_s", "stage", "outcome", "reason"])
    summ = dict(candidates=int(len(cand)), built=int(len(A["y"])), failed=int(len(F)), arrays=S.check_arrays(A), checks=dict(checks),
                built_by_shift={skey(s): int((M["shift_s"] == s).sum()) if len(M) else 0 for s in SHIFTS},
                failure_reasons={str(k): int(v) for k, v in F["reason"].value_counts().items()})
    return A, M, F, summ


def main(argv: list[str] | None = None) -> int:
    import argparse

    import psutil
    from football_fatigue.intersections.catalogue import load_steps
    from football_fatigue.matching import validate
    from football_fatigue.matching.benchmark import load_source
    from football_fatigue.window_intersections.__main__ import PROC, RAW, dumps, finish_manifest
    from football_fatigue.window_intersections.split import load_split

    ap = argparse.ArgumentParser(description="Stage 4C shift-augmentation samples (train/CV matches only)")
    ap.add_argument("--matches", nargs="+", default=["train_cv"])
    a = ap.parse_args(argv)
    sp = load_split(PROC, S.SPLIT)
    if S.split_sha256(PROC) != S.SPLIT_SHA256:
        raise SystemExit("frozen split hash differs from the approved split")
    matches = sorted(sp["train_cv"], key=int) if a.matches == ["train_cv"] else a.matches
    if set(matches) & set(sp["heldout"]) or not set(matches) <= set(sp["train_cv"]):
        raise SystemExit("only train/CV matches of the frozen split are allowed; held-out matches are never built")
    s2 = PROC / S.STAGE2
    s2man = json.loads((s2 / "manifest.json").read_text(encoding="utf-8"))
    s3 = PROC / "m11-stage3-samples-v1"
    ccfg = C.CandidateConfig()
    if json.dumps(ccfg.as_dict(), sort_keys=True) != json.dumps(s2man["candidate_config"], sort_keys=True):
        raise SystemExit("candidate configuration differs from Stage 2")
    out = out_dir()
    cfg = dumps(dict(schema=OUT, stage2_manifest_sha256=L._sha(s2 / "manifest.json"), stage3_manifest_sha256=L._sha(s3 / "manifest.json"),
                     split_sha256=S.split_sha256(PROC), matches=matches, gap_s=GAP, shifts_s=SHIFTS, candidate_config=ccfg.as_dict(),
                     stage3_code_sha256=S.code_sha256(), order_seed=S.ORDER_SEED, order_table=[table_name(s) for s in SHIFTS]))
    if out.exists() and (out / "manifest.json").exists():
        raise SystemExit(f"{out} is complete and immutable")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_config.json").exists():
        if (out / "run_config.json").read_text(encoding="utf-8") != cfg:
            raise SystemExit("run configuration differs from the interrupted run; refusing to mix")
    else:
        L.atomic_write_text(out / "run_config.json", cfg)
    proc = psutil.Process()
    snap0 = validate.raw_snapshot(RAW)
    t0 = time.perf_counter()

    def process(m: str, d: Path) -> dict[str, Any]:
        t = time.perf_counter()
        rel = f"matches/{m}/candidates.parquet"
        if L._sha(s2 / rel) != s2man["outputs_sha256"][rel]:
            raise RuntimeError(f"Stage 2 file {rel} does not match its manifest")
        L.verify_match_dir(s3 / "matches" / m)
        cand = pd.read_parquet(s2 / rel)
        if set(cand["split"]) != {"train_cv"}:
            raise RuntimeError(f"{m}: Stage 2 rows are not all train_cv")
        meta0 = pd.read_parquet(s3 / "matches" / m / "meta_g0p80.parquet")
        ms = pd.read_parquet(s3 / "matches" / m / "meta_g0p80_shift.parquet", columns=["candidate_id", "shift_s"])
        with np.load(s3 / "matches" / m / "samples_g0p80_shift.npz", allow_pickle=False) as z:
            s3_shift = {k: z[k] for k in S.ARRAY_FIELDS}
        s3_shift["candidate_id"] = ms["candidate_id"].to_numpy()
        s3_shift["shift_s"] = ms["shift_s"].to_numpy()
        c, mask, _e, _sm, _p = load_source(str(RAW), m)
        seg, _cnt, dup = W.build_segments(c, mask, load_steps(str(RAW), m))
        ch = W.chain_table(seg)
        A, M, F, summ = build_match_shifts(c, mask, dup, seg, ch, cand, meta0, s3_shift, ccfg, seed=int(m))
        S.save_npz(d / "samples.npz", A)
        L.atomic_parquet(M, d / "meta.parquet")
        L.atomic_parquet(F, d / "failures.parquet")
        summ["runtime_s"] = round(time.perf_counter() - t, 1)
        summ["rss_mb_after"] = round(proc.memory_info().rss / 2**20, 1)
        print(f"{m}: built {summ['built']} / {summ['candidates'] * len(SHIFTS)}  {summ['checks']}  {summ['runtime_s']} s", flush=True)
        return summ

    per = L.run_matches(out, matches, process)
    snap1 = validate.raw_snapshot(RAW)
    ok = all(r["arrays"]["all_pass"] and r["checks"].get("label_mismatch", 0) == 0 and r["checks"].get("perturbation_changed", 0) == 0
             and r["checks"].get("stage3_reproduction_different", 0) == 0 and r["checks"].get("gap_boundary_bad", 0) == 0 for r in per.values())
    tot = Counter()
    for r in per.values():
        tot.update(r["checks"])
        tot.update({f"built{k}": v for k, v in r["built_by_shift"].items()})
        tot.update({f"fail:{k}": v for k, v in r["failure_reasons"].items()})
    finish_manifest(out, dict(schema=OUT, matches=matches, heldout_built=False, all_checks_pass=bool(ok), raw_unchanged=snap0 == snap1,
                              totals=dict(tot), runtime_s=round(time.perf_counter() - t0, 1)))
    print(dumps(dict(all_checks_pass=ok, raw_unchanged=snap0 == snap1, totals=dict(tot))), flush=True)
    return 0 if ok and snap0 == snap1 else 1


def load(matches: list[str] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """(inputs, y, groups, meta) of the augmentation set; same refusal rules as the Stage 3 adapter. ``meta`` is evaluation /
    bookkeeping data (candidate id, shift, status) and never a model input."""
    from football_fatigue.window_intersections.__main__ import PROC
    from football_fatigue.window_intersections.split import load_split

    sp = load_split(PROC, S.SPLIT)
    if S.split_sha256(PROC) != S.SPLIT_SHA256:
        raise S.HeldOutRefused("the frozen split differs from the approved split")
    root = out_dir() / "matches"
    avail = sorted((p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith(".partial")), key=int)
    matches = avail if matches is None else [str(m) for m in matches]
    if set(matches) & set(sp["heldout"]) or set(avail) & set(sp["heldout"]) or not set(matches) <= set(sp["train_cv"]):
        raise S.HeldOutRefused("held-out matches are not readable during model development")
    xs, ys, gs, ms = [], [], [], []
    for m in matches:
        L.verify_match_dir(root / m)
        with np.load(root / m / "samples.npz", allow_pickle=False) as z:
            xs.append({k: z[k] for k in S.INPUT_FIELDS})
            ys.append(z["y"])
        gs.append(np.full(len(ys[-1]), int(m), np.int64))
        ms.append(pd.read_parquet(root / m / "meta.parquet"))
    return ({k: np.concatenate([x[k] for x in xs]) for k in S.INPUT_FIELDS}, np.concatenate(ys), np.concatenate(gs),
            pd.concat(ms, ignore_index=True))


if __name__ == "__main__":
    raise SystemExit(main())
