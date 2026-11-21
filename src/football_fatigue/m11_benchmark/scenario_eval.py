"""M11 Milestone 2: run and summarise the new-scenario suite (``scenarios``) with the Milestone 1 models and routing policy.

``python -m football_fatigue.m11_benchmark.scenario_eval run [--matches ...]``   (per match, written once, resumable)
``python -m football_fatigue.m11_benchmark.scenario_eval summarize``          (metrics + examples, written once)

Every scenario of match m is scored by the fold models that never saw m (``m11_infer.bundle.fold_for_match``). For every valid
scenario both raw model probabilities are stored (when the input has enough points), together with the routed operational
decision of policy ``m11-routing-v1`` and its outcome category:

- ``accepted_correct`` / ``accepted_wrong``: the routed model decided;
- ``abstain_low_confidence``: the routed model's confidence was below its threshold;
- ``abstain_unsupported``: the input was outside the policy (gap, declared noise or too few points); raw probabilities, when
  present, are diagnostic only.

The declared noise passed to the router is the noise added by the scenario (0 for clean provider data).

Outputs (``data/processed/m11-scenarios-v1``): ``matches/<m>/scenarios.parquet`` (one row per scenario, valid or invalid),
``matches/<m>/arrays.npz`` (trajectories for display: perturbed observations, grid tracklets in pitch coordinates, clean hidden
path of each player), ``summary/metrics_v1.json`` (metrics and the rule-based example sets under ``examples``), ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P
from football_fatigue.window_intersections import samples as S

from . import metrics as MT
from . import scenarios as SC

OUT = "m11-scenarios-v2"          # v1: aborted attempt (crashed on an empty tracklet, 11 matches, kept unchanged)
OPERATING_POINT = "error_05"
N_BOOT = 1000
OUTCOMES = ("accepted_correct", "accepted_wrong", "abstain_low_confidence", "abstain_unsupported")


def out_dir() -> Path:
    return B.proc_root() / OUT


# ---------------------------------------------------------------- routing of precomputed probabilities

def operational(p_gru: float | None, p_hgb: float | None, gap_s: float, noise_m: float | None, enough_points: bool, y: int | None = None,
                operating_point: str = OPERATING_POINT) -> dict[str, Any]:
    """The routed decision of ``predict.Predictor.predict_input`` from precomputed probabilities (same policy functions)."""
    r = P.route(gap_s, noise_m, enough_points)
    out: dict[str, Any] = dict(support=r.support, model=r.model, p_chosen=None, confidence=None, threshold=None, decision=P.ABSTAIN,
                               predicted_pairing=None, abstain_reason="", reasons=" ; ".join(r.reasons))
    if r.model is None:
        out["abstain_reason"] = r.reasons[0].split(":")[0]
        out["outcome"] = "abstain_unsupported"
        return out
    p = p_gru if r.model == "gru_c" else p_hgb
    th = B.THRESHOLDS[r.model][operating_point]
    pairing, dec, abst, conf = P.decide(p, th)
    out.update(p_chosen=p, confidence=conf, threshold=th, decision=dec, predicted_pairing=pairing)
    if abst:
        out["abstain_reason"] = "low_confidence"
        out["outcome"] = "abstain_low_confidence"
    else:
        correct = (pairing == P.SWAPPED) == bool(y) if y is not None else None
        out["outcome"] = "accepted_correct" if correct else "accepted_wrong"
    return out


# ---------------------------------------------------------------- one match

_MODELS: dict[int, dict[str, Any]] = {}


def fold_models(fold: int) -> dict[str, Any]:
    if fold not in _MODELS:
        _MODELS[fold] = {m: B.load_fold_model(m, fold, device="cpu") for m in B.MODELS}
    return _MODELS[fold]


def evaluate_match(match_id: str, n_per_match: int = SC.N_PER_MATCH, log=print) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    t0 = time.perf_counter()
    md = SC.MatchData(match_id)
    models = fold_models(md.fold)
    enc = SC.select_encounters(md.meta, md.match_id, n_per_match)
    rows, mis, scs = [], [], []
    for _, e in enc.iterrows():
        fam = SC.encounter_families(md, e)
        for factor, level in SC.VARIANTS:
            sc = SC.build(md, e, factor, level, fam)
            row = dict(scenario_id=sc.scenario_id, match_id=md.match_id, fold=md.fold, candidate_id=sc.candidate_id, factor=factor,
                       level=sc.level, gap_param_s=sc.params["gap"], shift_s=sc.params["shift"], noise_m=sc.params["noise"],
                       dropout=sc.params["dropout"], context_s=sc.params["context"], valid=sc.valid, invalid_reason=sc.invalid_reason,
                       label_source=sc.label_source, **{f"fam_{k}": v for k, v in fam.items()})
            if sc.valid:
                mi = model_input(sc)
                row.update(y=sc.y, g0=sc.g0, g1=sc.g1, tc=sc.tc, gap_s=mi.gap_s, who_in=",".join(sc.who_in), who_out=",".join(sc.who_out),
                           n_valid_min=int(min(mi.n_valid)), n_valid=",".join(map(str, mi.n_valid)), enough_points=mi.enough_points,
                           frame_mode=-1 if mi.frame_mode is None else int(mi.frame_mode), arr_row=len(mis))
                mis.append(mi)
                scs.append(sc)
            rows.append(row)
    df = pd.DataFrame(rows)
    # batched raw predictions (identical arithmetic to Predictor.predict_arrays; one batch per model)
    ok = [i for i, mi in enumerate(mis) if mi.enough_points]
    pg, ph = np.full(len(mis), np.nan), np.full(len(mis), np.nan)
    if ok:
        X = {k: np.concatenate([mis[i].arrays[k] for i in ok]) for k in S.INPUT_FIELDS}
        pg[ok], ph[ok] = models["gru_c"].predict(X), models["hgb_c"].predict(X)
    v = df["valid"].to_numpy(bool)
    df["p_gru"], df["p_hgb"] = np.nan, np.nan
    df.loc[v, "p_gru"], df.loc[v, "p_hgb"] = pg, ph
    ops = []
    for _, r in df.iterrows():
        if not r["valid"]:
            ops.append(dict(outcome="invalid"))
            continue
        ops.append(operational(None if np.isnan(r["p_gru"]) else float(r["p_gru"]), None if np.isnan(r["p_hgb"]) else float(r["p_hgb"]),
                               float(r["gap_s"]), float(r["noise_m"]), bool(r["enough_points"]), int(r["y"])))
    df = pd.concat([df, pd.DataFrame(ops)], axis=1)
    df["gru_checkpoint_sha256"] = models["gru_c"].sha256
    df["hgb_checkpoint_sha256"] = models["hgb_c"].sha256
    df["policy"] = P.POLICY
    arrays = _arrays(mis, scs)
    info = dict(match_id=md.match_id, fold=md.fold, encounters=int(len(enc)), scenarios=int(len(df)), valid=int(v.sum()),
                outcomes={k: int(c) for k, c in Counter(df["outcome"]).items()}, runtime_s=round(time.perf_counter() - t0, 1))
    log(f"{md.match_id} (fold {md.fold}): {info['encounters']} encounters, {info['scenarios']} scenarios, {info['valid']} valid  "
        f"{info['outcomes']}  {info['runtime_s']} s")
    return df, arrays, info


def model_input(sc: SC.Scenario) -> A.ModelInput:
    """Adapter input of a valid scenario. A perturbation can remove every observation of a tracklet; that input has no valid grid
    point, so it is returned with all masks false (and routed as unsupported: too few valid points). Nothing is filled in."""
    if all(len(tr.t) for tr in (*sc.incoming, *sc.outgoing)):
        return A.build_inputs(sc.incoming, sc.outgoing, sc.g0, sc.g1)
    z2, zm = np.zeros((2, S.T, 2)), np.zeros((2, S.T), bool)
    nv = tuple(0 if not len(tr.t) else int(len(tr.t)) for tr in (*sc.incoming, *sc.outgoing))
    return A.ModelInput(arrays={}, g0=sc.g0, g1=sc.g1, grid_in=sc.g0 + S.IN_REL, grid_out=sc.g1 + S.OUT_REL, pitch_in=z2, pitch_out=z2.copy(),
                        in_mask=zm, out_mask=zm.copy(), n_valid=tuple(0 for _ in nv), origin=None, R=None, frame_mode=None,
                        ignored_in_gap=(0, 0, 0, 0), notes=["a tracklet has no observations after the perturbation"])


def _arrays(mis: list[A.ModelInput], scs: list[SC.Scenario]) -> dict[str, np.ndarray]:
    n = len(mis)
    a = dict(pitch_in=np.stack([m.pitch_in for m in mis]).astype(np.float32) if n else np.zeros((0, 2, S.T, 2), np.float32),
             in_mask=np.stack([m.in_mask for m in mis]) if n else np.zeros((0, 2, S.T), bool),
             pitch_out=np.stack([m.pitch_out for m in mis]).astype(np.float32) if n else np.zeros((0, 2, S.T, 2), np.float32),
             out_mask=np.stack([m.out_mask for m in mis]) if n else np.zeros((0, 2, S.T), bool),
             hidden=np.stack([s.hidden for s in scs]) if n else np.zeros((0, 2, SC.HIDDEN_LEN, 2), np.float32),
             hidden_mask=np.stack([s.hidden_mask for s in scs]) if n else np.zeros((0, 2, SC.HIDDEN_LEN), bool))
    ts, xs, ys, off, k = [], [], [], np.zeros((n, 4, 2), np.int64), 0
    for i, s in enumerate(scs):
        for j, tr in enumerate((*s.incoming, *s.outgoing)):
            off[i, j] = (k, k + len(tr.t))
            k += len(tr.t)
            ts.append(tr.t)
            xs.append(tr.x)
            ys.append(tr.y)
    a.update(obs_t=np.concatenate(ts) if ts else np.zeros(0), obs_x=np.concatenate(xs) if xs else np.zeros(0),
             obs_y=np.concatenate(ys) if ys else np.zeros(0), obs_offsets=off)
    return a


def _log(msg: str) -> None:
    print(msg, flush=True)


def run(matches: list[str] | None = None, n_per_match: int = SC.N_PER_MATCH, log=_log) -> None:
    from football_fatigue.m11_benchmark import data as D
    from football_fatigue.window_intersections import local as L

    out = out_dir()
    matches = B.train_cv_matches() if not matches else [str(int(m)) for m in matches]
    for m in matches:
        B.fold_for_match(m)
    cfg = dict(schema=OUT, suite=SC.SUITE, n_per_match=n_per_match, base=SC.BASE, factors=SC.FACTORS, policy=P.POLICY,
               operating_point=OPERATING_POINT, thresholds=B.THRESHOLDS, split_sha256=S.split_sha256(B.proc_root()),
               bundles_manifest_sha256=B.sha256_file(B.manifest_path()),
               code_sha256={p.name: B.sha256_file(p) for p in (Path(SC.__file__), Path(__file__), Path(P.__file__), Path(A.__file__),
                                                               Path(B.__file__))})
    out.mkdir(parents=True, exist_ok=True)
    rc = out / "run_config.json"
    if rc.exists() and json.loads(rc.read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        raise SystemExit(f"{rc} differs from this configuration; outputs are never mixed or overwritten")
    B.write_json_once(rc, cfg)
    for m in matches:
        d = out / "matches" / m
        if (d / "done.json").exists():
            log(f"{m}: done (skip)")
            continue
        df, arrays, info = evaluate_match(m, n_per_match, log)
        d.mkdir(parents=True, exist_ok=True)
        L.atomic_parquet(df, d / "scenarios.parquet")
        info["arrays_sha256"] = D.write_once_npz(d / "arrays.npz", arrays)
        info["scenarios_sha256"] = B.sha256_file(d / "scenarios.parquet")
        B.write_json_once(d / "done.json", info)


def load_results(out: Path | None = None) -> pd.DataFrame:
    out = out or out_dir()
    parts = [pd.read_parquet(p) for p in sorted((out / "matches").glob("*/scenarios.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_arrays(match_id: str) -> dict[str, np.ndarray]:
    from football_fatigue.m11_benchmark import data as D
    return D.load_npz(out_dir() / "matches" / str(match_id) / "arrays.npz")


# ---------------------------------------------------------------- summary

def _raw(df: pd.DataFrame, col: str, boot: bool = True) -> dict[str, Any]:
    d = df[np.isfinite(df[col])]
    if not len(d):
        return dict(n=0)
    y, p, g = d["y"].to_numpy(int), d[col].to_numpy(float), d["match_id"].to_numpy()
    r = MT.basic(y, p)
    r.pop("precision_swapped", None)
    r.pop("recall_swapped", None)
    if boot and len(np.unique(g)) > 1 and 0 < y.sum() < len(y):
        r["ba_ci"] = MT.bootstrap(y, p, g, n_boot=N_BOOT)["balanced_accuracy_ci"]
    return r


def _routed(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    c = Counter(df["outcome"])
    acc = c["accepted_correct"] + c["accepted_wrong"]
    sub = df[df["outcome"] == "abstain_unsupported"]
    return dict(n=n, **{k: int(c[k]) for k in OUTCOMES}, coverage=MT._r(acc / n) if n else None,
                realized_error=MT._r(c["accepted_wrong"] / acc) if acc else None,
                support={k: int(v) for k, v in Counter(df["support"]).items()},
                model_used={str(k): int(v) for k, v in Counter(df["model"].fillna("none")).items()},
                unsupported_reasons={k: int(v) for k, v in Counter(sub["abstain_reason"]).items()})


def _reliability(df: pd.DataFrame, col: str, bins: int = 10) -> list[dict[str, Any]]:
    d = df[np.isfinite(df[col])]
    y, p = d["y"].to_numpy(int), d[col].to_numpy(float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [dict(bin=k, p_mean=MT._r(p[b == k].mean()), y_rate=MT._r(y[b == k].mean()), n=int((b == k).sum())) for k in range(bins) if (b == k).any()]


FAMILY_COLS = ("fam_heading_relation", "fam_distance_trend", "fam_heading_change", "fam_speed_change", "fam_hard_parallel",
               "fam_crossing_class", "fam_team_relation", "fam_d_band", "fam_speed_band", "fam_status_in_gap")
FAMILY_LEVELS = (("baseline", "base"), ("gap", "1.6"), ("noise", "0.05"), ("noise", "0.25"))


def summarize(log=print) -> dict[str, Any]:
    df = load_results()
    if not len(df):
        raise SystemExit("no scenario results")
    v = df[df["valid"]]
    rep: dict[str, Any] = dict(schema="m11-scenarios-metrics-v1", suite=SC.SUITE, policy=P.POLICY, operating_point=OPERATING_POINT,
                               n_boot=N_BOOT, matches=int(df["match_id"].nunique()), encounters=int(df["candidate_id"].nunique()),
                               scenarios=int(len(df)), valid=int(len(v)), invalid=int((~df["valid"]).sum()),
                               note="development stress cases: synthetic interruptions of real trajectories in the 32 train/CV "
                                    "matches, scored by unseen-match fold models; not independent matches or real occlusions")
    base = v[v["factor"] == "baseline"]
    rep["by_factor"] = {}
    for factor in SC.FACTORS:
        levels = [("base", base)] + [(SC.level_key(lv), v[(v["factor"] == factor) & (v["level"] == SC.level_key(lv))]) for lv in SC.FACTORS[factor]]
        tab = {}
        for key, d in levels:
            allr = df[(df["factor"] == ("baseline" if key == "base" else factor)) & (df["level"] == key)]
            e = dict(encounters=int(len(allr)), valid=int(len(d)), invalid_reasons=_top(allr[~allr["valid"]]["invalid_reason"]),
                     enough_points=int(d["enough_points"].sum()) if len(d) else 0,
                     gru=_raw(d, "p_gru"), hgb=_raw(d, "p_hgb"), routed=_routed(d))
            dd = d[np.isfinite(d["p_gru"]) & np.isfinite(d["p_hgb"])]
            if len(dd) and dd["match_id"].nunique() > 1 and 0 < dd["y"].sum() < len(dd):
                e["gru_minus_hgb"] = MT.paired(dd["y"].to_numpy(int), dd["p_gru"].to_numpy(), dd["p_hgb"].to_numpy(), dd["match_id"].to_numpy(),
                                               n_boot=N_BOOT)
            tab[key] = e
        rep["by_factor"][factor] = tab
        log(f"{factor}: " + "  ".join(f"{k}: GRU {t['gru'].get('balanced_accuracy')} HGB {t['hgb'].get('balanced_accuracy')} "
                                      f"cov {t['routed']['coverage']} err {t['routed']['realized_error']}" for k, t in tab.items()))
    # gap factor on the encounters valid at every gap level (removes the selection effect of label re-validation)
    g = v[(v["factor"] == "gap") | (v["factor"] == "baseline")]
    cnt = g.groupby("candidate_id")["scenario_id"].count()
    common = set(cnt[cnt == len(SC.FACTORS["gap"]) + 1].index)
    rep["gap_common_subset"] = dict(encounters=len(common), levels={})
    for key in ["base"] + [SC.level_key(x) for x in SC.FACTORS["gap"]]:
        d = g[g["candidate_id"].isin(common) & (g["level"] == key)]
        rep["gap_common_subset"]["levels"][key] = dict(gru=_raw(d, "p_gru", boot=False), hgb=_raw(d, "p_hgb", boot=False), routed=_routed(d))
    rep["families"] = {}
    for factor, key in FAMILY_LEVELS:
        d = v[(v["factor"] == factor) & (v["level"] == key)]
        rep["families"][f"{factor}={key}"] = {c[4:]: {str(val): dict(n=int(len(x)), gru=_raw(x, "p_gru", boot=False), hgb=_raw(x, "p_hgb", boot=False),
                                                                     routed=_routed(x))
                                                      for val, x in d.groupby(c)} for c in FAMILY_COLS}
    rep["reliability"] = {f"{f}={k}": {m: _reliability(v[(v["factor"] == f) & (v["level"] == k)], f"p_{m}") for m in ("gru", "hgb")}
                          for f, k in FAMILY_LEVELS + (("gap", "2.4"), ("shift", "0.5"))}
    rep["overall_routed"] = _routed(v)
    rep["examples"] = examples(v)
    return rep


def _top(s: pd.Series, k: int = 5) -> dict[str, int]:
    c = Counter(r.split(":")[0] + (":" + r.split(":")[1].split("|")[0] if ":" in r else "") for r in s)
    return dict(c.most_common(k))


def example_sets(v: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The rule-based example categories over the valid scenarios with both probabilities (also used by the dashboard)."""
    p = v[np.isfinite(v["p_gru"]) & np.isfinite(v["p_hgb"])].copy()
    p["gru_ok"] = (p["p_gru"] > 0.5).astype(int) == p["y"]
    p["hgb_ok"] = (p["p_hgb"] > 0.5).astype(int) == p["y"]
    chosen_wrong = ((p["p_chosen"] > 0.5).astype(int) != p["y"])
    sets = {
        "gru_correct_primary": p[(p["model"] == "gru_c") & (p["outcome"] == "accepted_correct")],
        "gru_wrong_primary": p[(p["model"] == "gru_c") & (p["outcome"] == "accepted_wrong")],
        "hgb_correct_fallback": p[(p["model"] == "hgb_c") & (p["outcome"] == "accepted_correct")],
        "hgb_wrong_fallback": p[(p["model"] == "hgb_c") & (p["outcome"] == "accepted_wrong")],
        "models_disagree": p[(p["p_gru"] > 0.5) != (p["p_hgb"] > 0.5)],
        "successful_abstention": p[(p["outcome"] == "abstain_low_confidence") & chosen_wrong],
        "abstention_of_correct": p[(p["outcome"] == "abstain_low_confidence") & ~chosen_wrong],
        "long_gap_ge_1p6": p[(p["factor"] == "gap") & (p["gap_param_s"] >= 1.6)],
        "noisy_ge_10cm": p[(p["factor"] == "noise") & (p["noise_m"] >= 0.10)],
        "hard_parallel_baseline": p[(p["factor"] == "baseline") & p["fam_hard_parallel"].astype(bool)],
    }
    return p, sets


def examples(v: pd.DataFrame, k: int = 6) -> dict[str, Any]:
    """Rule-based example sets (no cherry-picking): within each category, scenarios in scenario-id order (a hash), except the
    explicitly named 'highest_confidence' sets, which are ordered by confidence."""
    p, sets = example_sets(v)
    out = {name: dict(n_in_category=int(len(d)), scenario_ids=d.sort_values("scenario_id")["scenario_id"].head(k).tolist()) for name, d in sets.items()}
    hc = p[p["outcome"] == "accepted_wrong"].sort_values(["confidence", "scenario_id"], ascending=[False, True])
    out["highest_confidence_accepted_errors"] = dict(n_in_category=int(len(hc)), scenario_ids=hc["scenario_id"].head(k).tolist(),
                                                     note="ordered by confidence (largest first), not a random sample")
    for d in out.values():
        if "scenario_ids" in d:
            sel = p.set_index("scenario_id").loc[d["scenario_ids"]]
            d["rows"] = [dict(scenario_id=i, match_id=r["match_id"], candidate_id=r["candidate_id"], factor=r["factor"], level=r["level"],
                              y=int(r["y"]), p_gru=MT._r(r["p_gru"]), p_hgb=MT._r(r["p_hgb"]), decision=r["decision"], outcome=r["outcome"],
                              support=r["support"], model=r["model"]) for i, r in sel.iterrows()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_benchmark.scenario_eval")
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--matches", nargs="+")
    ap.add_argument("--n-per-match", type=int, default=SC.N_PER_MATCH)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        t = time.perf_counter()
        run(a.matches, a.n_per_match)
        print(f"run finished in {time.perf_counter() - t:.0f} s", flush=True)
        return 0
    rep = summarize()
    B.write_json_once(out_dir() / "summary" / "metrics_v1.json", rep)
    man = dict(schema=OUT, matches=sorted(load_results()["match_id"].unique().tolist(), key=int),
               per_match={p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir() / "matches").glob("*/done.json"))},
               metrics_sha256=B.sha256_file(out_dir() / "summary" / "metrics_v1.json"), heldout_used=False, official_data_used=False)
    B.write_json_once(out_dir() / "manifest.json", man)
    print(f"wrote {out_dir() / 'summary' / 'metrics_v1.json'} and manifest", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
