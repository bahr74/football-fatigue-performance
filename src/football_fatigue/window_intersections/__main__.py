"""Build the M11 window-intersection datasets (shared layer + methods A, B, C) on FIFA development matches.

    python -m football_fatigue.window_intersections --layout pilot --matches 3812 3816 --workers 2
    python -m football_fatigue.window_intersections --layout full --workers 4            # all 40 development matches

Outputs are new versioned directories under ``data/processed``; an existing directory is never overwritten. RAW data is read only
(a size/mtime digest is taken before and after). Validation and test matches are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.intersections.catalogue import load_steps
from football_fatigue.matching import validate
from football_fatigue.matching.benchmark import load_source

from . import core as W

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SPLITS = PROC / "m11-encounters-v1" / "splits.json"
METHODS = {
    "A": dict(slug="a", title="Method A: geometric intersection + temporal tolerance |tA - tB| <= tau",
              full_dir="m11-window-method-a-intersection-time-v1"),
    "B": dict(slug="b", title="Method B: geometric intersection + anchor-time synchronized distance <= R",
              full_dir="m11-window-method-b-intersection-anchor-distance-v1"),
    "C": dict(slug="c", title="Method C: geometric intersection + temporal AND anchor-distance tolerances",
              full_dir="m11-window-method-c-intersection-time-and-distance-v1"),
}
SHARED_FULL = "m11-window-intersections-v1"
PILOT = "m11-window-intersections-pilot-v1"
EXAMPLE_MATCH_DEFAULT = "3812"


def dev_matches() -> list[str]:
    s = json.loads(SPLITS.read_text(encoding="utf-8"))
    return sorted((m for m, v in s["matches"].items() if v["split"] == "development"), key=int)


def layout_dirs(layout: str) -> tuple[Path, dict[str, Path]]:
    if layout == "pilot":
        root = PROC / PILOT
        return root / "shared", {k: root / f"method_{v['slug']}" for k, v in METHODS.items()}
    return PROC / SHARED_FULL, {k: PROC / v["full_dir"] for k, v in METHODS.items()}


def code_sha256() -> dict[str, str]:
    files = sorted(Path(__file__).parent.glob("*.py"))
    base = Path(__file__).parents[1]
    files += [base / "intersections" / n for n in ("exact.py", "detect.py", "geometry.py", "catalogue.py")]
    files += [base / "encounters" / "quality.py", base / "matching" / "benchmark.py"]
    return {str(p.relative_to(base)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _clean(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def dumps(o: Any) -> str:
    return json.dumps(_clean(o), indent=1, sort_keys=True)


# ---------------------------------------------------------------- per match (worker)

def match_checks(c, seg, res: dict[str, dict], cfg: W.WindowConfig, brute_n: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v, r in res.items():
        dets, evs = r["detections"], r["events"]
        keys = [(d["period"], d["anchor_s"], d["slot_a"], d["slot_b"]) for d in dets]
        chk = dict(pair_window_keys_unique=len(set(keys)) == len(keys), slot_order_a_lt_b=all(d["slot_a"] < d["slot_b"] for d in dets),
                   detections_equal_sum_event_windows=len(dets) == sum(e["n_windows"] for e in evs),
                   every_detection_maps_to_one_event=len({d["event_id"] for d in dets}) == len(evs))
        chk["c_equals_a_and_b"] = all(e[f"c_tau_{cfg.tau_key(t)}"] == (e[f"a_tau_{cfg.tau_key(t)}"] and e["b_pass"]) for e in evs for t in cfg.taus_s)
        ks = [cfg.tau_key(t) for t in sorted(cfg.taus_s)]
        chk["a_nested_in_tau"] = all(not e[f"a_tau_{ks[i]}"] or e[f"a_tau_{ks[i + 1]}"] for e in evs for i in range(len(ks) - 1))
        chk["event_ids_unique"] = len({e["event_id"] for e in evs}) == len(evs)
        out[v] = chk
    # PCHIP stays inside each interval's sample bounding box and passes through the samples
    cx, cy, sup = res["pchip"]["cx"], res["pchip"]["cy"], res["pchip"]["sup"]
    rng = np.random.default_rng(seed)
    g = np.flatnonzero(sup)
    g = rng.choice(g, min(len(g), 50_000), replace=False) if len(g) else g
    px, py, _ = W.sub_points(seg, cx, cy, g, 40)
    s = seg["dt"][g]
    vx = np.abs(W.evalc(cx[g], s) - seg["x1"][g]).max() if len(g) else 0.0
    ex = np.maximum.reduce([px.max(1) - np.maximum(seg["x0"][g], seg["x1"][g]), np.minimum(seg["x0"][g], seg["x1"][g]) - px.min(1),
                            py.max(1) - np.maximum(seg["y0"][g], seg["y1"][g]), np.minimum(seg["y0"][g], seg["y1"][g]) - py.min(1)]) if len(g) else np.zeros(1)
    out["pchip"]["segments_checked"] = int(len(g))
    out["pchip"]["max_excursion_outside_sample_bbox_m"] = float(max(ex.max(), 0.0))
    out["pchip"]["max_vertex_mismatch_m"] = float(vx)
    out["pchip"]["segments_pchip_supported"] = int(sup.sum())
    out["pchip"]["segments_pchip_unsupported"] = int((~sup).sum())
    if brute_n:
        out["brute_force"] = brute(c, seg, res, cfg, brute_n, seed)
    return out


def brute(c, seg, res, cfg: W.WindowConfig, n: int, seed: int) -> dict[str, Any]:
    """Independent re-computation: clip both trajectories to the window and intersect all piece pairs with
    ``intersections.geometry.intersect``. Half the samples are detected pair-windows, half random active pairs."""
    rng = np.random.default_rng(seed + 1)
    out = {}
    for v, r in res.items():
        K = cfg.pchip_substeps if v == "pchip" else 1
        det = {(d["period"], d["anchor_s"], d["slot_a"], d["slot_b"]): d for d in r["detections"]}
        per = {p["period"]: p for p in r["scan"]["periods"]}
        samples = [k for k in rng.permutation(list(det))[: n // 2]]
        samples = [tuple(int(x) for x in k) for k in samples]
        while len(samples) < n:
            p = int(rng.choice(list(per)))
            st, en = per[p]["start_s"], per[p]["end_s"]
            k = int(rng.integers(3, int(en - st - 3)))
            sa, sb = sorted(int(x) for x in rng.choice(np.unique(seg["slot"][seg["period"] == p]), 2, replace=False))
            samples.append((p, k, sa, sb))
        mism, n_pos, n_cmp = [], 0, 0
        for (p, k, sa, sb) in samples:
            st = per[p]["start_s"]
            got = W.brute_force_window(seg, r["cx"], r["cy"], sa, sb, p, st + k - cfg.half_s, st + k + cfg.half_s, K, cfg)
            d = det.get((p, k, sa, sb))
            n_cmp += 1
            if d is None:
                if got:
                    mism.append(dict(key=[p, k, sa, sb], fast=None, brute_n=len(got)))
                continue
            n_pos += 1
            h = r["hits"][p]
            fa, fb = float(h["ta"][d["hit"]]), float(h["tb"][d["hit"]])
            if not got or len(got) != d["n_intersections"] or abs(got[0][0] - fa) > 1e-6 or abs(got[0][1] - fb) > 1e-6:
                mism.append(dict(key=[p, k, sa, sb], fast=[fa, fb, d["n_intersections"]], brute=[got[0] if got else None, len(got)]))
        out[v] = dict(pair_windows_compared=n_cmp, with_detection=n_pos, mismatches=len(mism), mismatch_examples=mism[:10])
    return out


def process_match(args: tuple) -> dict[str, Any]:
    m, shared, cfg_d, brute_n = args
    s = json.loads(SPLITS.read_text(encoding="utf-8"))
    if s["matches"][m]["split"] != "development":
        raise SystemExit(f"match {m} is not a development match; refused")
    cfg = W.WindowConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_d.items()})
    t0 = time.perf_counter()
    c, mask, _enc, src_manifest, prov = load_source(str(RAW), m)
    steps = load_steps(str(RAW), m)
    seg, seg_counts, dup = W.build_segments(c, mask, steps)
    ch = W.chain_table(seg)
    res = {v: W.run_variant(c, mask, seg, ch, dup, v, cfg) for v in W.VARIANTS}
    cmp = W.compare_variants(res["linear"]["events"], res["pchip"]["events"])
    checks = match_checks(c, seg, res, cfg, brute_n, int(m))
    out = Path(shared) / m
    out.mkdir(parents=True, exist_ok=False)
    for v, r in res.items():
        pd.DataFrame(r["events"]).to_parquet(out / f"events_{v}.parquet", index=False)
        dd = pd.DataFrame(r["detections"])
        dd["extra"] = dd["extra"].map(json.dumps)
        dd.to_parquet(out / f"pair_windows_{v}.parquet", index=False)
    summary = dict(schema=W.SCHEMA, match_id=int(m), split="development", segments=seg_counts, duplicate_position_samples=int(dup.sum()),
                   timestamps=dict(periods={int(p): dict(start_s=pp["start_s"], end_s=pp["end_s"]) for v in ["linear"] for pp in res[v]["scan"]["periods"] for p in [pp["period"]]},
                                   strictly_increasing=True),
                   scan={v: {k: x for k, x in r["scan"].items()} for v, r in res.items()}, linear_vs_pchip=cmp, checks=checks,
                   source=dict(step1=prov, step1_schema=src_manifest["schema"]), seconds=round(time.perf_counter() - t0, 1))
    (out / "match_summary.json").write_text(dumps(summary), encoding="utf-8")
    det_cols = ["match_id", "variant", "period", "anchor_s", "slot_a", "slot_b", "event_id", "n_intersections", "abs_dt_s", "anchor_dist_m",
                "window_min_raw_dist_m", "complete_a", "complete_b"]
    return dict(match=m, summary=summary, events={v: pd.DataFrame(r["events"]) for v, r in res.items()},
                detections={v: pd.DataFrame(r["detections"])[det_cols] for v, r in res.items()})


# ---------------------------------------------------------------- method datasets

def method_masks(M: str, ev: pd.DataFrame, det: pd.DataFrame, cfg: W.WindowConfig) -> list[tuple[str, pd.Series, pd.Series, pd.Series]]:
    """(threshold label, event pass, detection pass, passing-window count per event) for every configured threshold."""
    out = []
    if M in ("A", "C"):
        for tau in cfg.taus_s:
            k = cfg.tau_key(tau)
            if M == "A":
                e, d = ev[f"a_tau_{k}"], det["abs_dt_s"] <= tau + 1e-12
                nw = ev["n_windows"]
                out.append((f"tau={tau:.2f}s", e, d, nw))
            else:
                e = ev[f"c_tau_{k}"]
                d = (det["abs_dt_s"] <= tau + 1e-12) & (det["anchor_dist_m"] <= cfg.radius_m)
                nw = ev["b_pass_windows"].map(lambda s: len(json.loads(s)))
                out.append((f"tau={tau:.2f}s & R={cfg.radius_m:.2f}m", e, d, nw))
    else:
        out.append((f"R={cfg.radius_m:.2f}m", ev["b_pass"], det["anchor_dist_m"] <= cfg.radius_m, ev["b_pass_windows"].map(lambda s: len(json.loads(s)))))
    return out


def _explode(series: pd.Series) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for s in series:
        for r in json.loads(s):
            r = r[:-2] if r.endswith(("_a", "_b")) else r
            cnt[r] = cnt.get(r, 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


def _q(v: pd.Series) -> dict[str, float] | None:
    v = pd.to_numeric(v, errors="coerce").dropna()
    return None if v.empty else {f"p{q}": round(float(np.percentile(v, q)), 4) for q in (5, 25, 50, 75, 95)} | {"n": int(len(v))}


def method_counts(M: str, ev: pd.DataFrame, det: pd.DataFrame, cfg: W.WindowConfig) -> dict[str, Any]:
    res = {}
    for label, e, d, nw in method_masks(M, ev, det, cfg):
        E = ev[e]
        n_det, n_ev = int(d.sum()), int(e.sum())
        if set(det.loc[d, "event_id"]) != set(E["event_id"]):
            raise AssertionError(f"{M} {label}: detection-level and event-level selections disagree")
        ctx = dict(context_full_2_5s_both=int(E["context_full"].sum()), context_min_1s_both=int(E["context_min"].sum()),
                   sample_ready=int(E["sample_ready"].sum()), label_reliable=int(E["label_reliable"].sum()))
        res[label] = dict(
            raw_pair_window_detections=n_det, duplicates_removed=n_det - n_ev, events_merged_from_multiple_windows=int((nw[e] > 1).sum()),
            independent_events=n_ev, distinct_encounters=int(E["encounter_id"].nunique()),
            unique_player_pairs=int(E[["match_id", "slot_a", "slot_b"]].drop_duplicates().shape[0]),
            pair_windows_with_multiple_intersections=int((det.loc[d, "n_intersections"] > 1).sum()),
            status={k: int(v) for k, v in E["status"].value_counts().items()}, tier={k: int(v) for k, v in E["tier"].value_counts().items()},
            valid_after_cleaning=int((E["status"] != "rejected").sum()), accepted=int((E["status"] == "accepted").sum()),
            context=ctx, third_player_within_2m=int((E["crowd_others_within_2m"] > 0).sum()),
            shared_player_overlap=int((E["shared_player_events"] > 0).sum()),
            per_match_events=_q(E.groupby("match_id").size()), abs_dt_s=_q(E["abs_dt_s"]), anchor_dist_min_m=_q(E["anchor_dist_min_m"]),
            reject_reasons=_explode(E["reject_reasons"]), flag_reasons=_explode(E["flag_reasons"]), info_flags=_explode(E["info_flags"]),
            before_after_s={f"{w}_{t}": _q(E[f"{w}_s_{t}"]) for w in ("before", "after") for t in ("a", "b")})
    return res


def threshold_rejections(M: str, ev: pd.DataFrame, cfg: W.WindowConfig) -> dict[str, int]:
    """Primary-intersection events that fail the method's condition, by reason (loosest threshold)."""
    tmax = max(cfg.taus_s)
    fa = ev["abs_dt_s"] > tmax + 1e-12
    unavailable = ~np.isfinite(ev["anchor_dist_min_m"].astype(float))
    fb = ~ev["b_pass"]
    out = dict(primary_events_total=int(len(ev)))
    if M == "A":
        out[f"abs_dt_gt_{tmax:.2f}s"] = int(fa.sum())
    if M in ("B", "C"):
        out[f"anchor_dist_gt_{cfg.radius_m:.2f}m"] = int((fb & ~unavailable).sum())
        out["anchor_position_unavailable_all_windows"] = int((fb & unavailable).sum())
    if M == "C":
        out["fails_time_only (passes R)"] = int((fa & ~fb).sum())
        out[f"fails_distance_only (passes tau={tmax:.2f})"] = int((~fa & fb).sum())
        out["fails_both"] = int((fa & fb).sum())
    out["passes"] = int(ev[pass_col(M, cfg)].sum())
    return out


def pass_col(M: str, cfg: W.WindowConfig) -> str:
    return {"A": f"a_tau_{cfg.tau_key(max(cfg.taus_s))}", "B": "b_pass", "C": f"c_tau_{cfg.tau_key(max(cfg.taus_s))}"}[M]


def variant_comparison(M: str, evs: dict[str, pd.DataFrame], cfg: W.WindowConfig) -> dict[str, Any]:
    col = pass_col(M, cfg)
    L, P = evs["linear"], evs["pchip"]
    pmap = P.set_index("event_id")[col].to_dict()
    lp = L[L[col]]
    matched = lp["pchip_match"].notna()
    both = lp[matched]["pchip_match"].map(pmap).fillna(False).astype(bool)
    pp = P[P[col]]
    return dict(linear_pass=int(len(lp)), pchip_pass=int(len(pp)),
                linear_pass_with_pchip_event=int(matched.sum()), linear_pass_pchip_event_also_passes=int(both.sum()),
                linear_pass_pchip_event_fails=int((~both).sum()), linear_pass_no_pchip_event=int((~matched).sum()),
                pchip_pass_no_linear_event=int(pp["linear_match"].isna().sum()),
                pchip_pass_linear_event_fails=int((pp["linear_match"].notna() & ~pp["linear_match"].map(L.set_index("event_id")[col].to_dict()).fillna(False).astype(bool)).sum()),
                location_diff_m_matched=_q(lp.loc[matched, "pchip_loc_diff_m"]), abs_dt_change_s_matched=_q(lp.loc[matched, "pchip_abs_dt_change_s"]),
                status_pchip={k: int(v) for k, v in pp["status"].value_counts().items()})


def write_method(M: str, out: Path, evs: dict[str, pd.DataFrame], dets: dict[str, pd.DataFrame], cfg: W.WindowConfig, matches: list[str],
                 code: dict, layout: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    col = pass_col(M, cfg)
    counts = {v: method_counts(M, evs[v], dets[v], cfg) for v in W.VARIANTS}
    rej = {v: dict(threshold=threshold_rejections(M, evs[v], cfg),
                   quality_of_passing=dict(status={k: int(x) for k, x in evs[v].loc[evs[v][col], "status"].value_counts().items()},
                                           reject_reasons=_explode(evs[v].loc[evs[v][col], "reject_reasons"]),
                                           flag_reasons=_explode(evs[v].loc[evs[v][col], "flag_reasons"]))) for v in W.VARIANTS}
    comp = variant_comparison(M, evs, cfg)
    for v in W.VARIANTS:
        E = evs[v][evs[v][col]].sort_values(["match_id", "period", "t_event_s", "slot_a", "slot_b"]).reset_index(drop=True)
        E.insert(0, "method", M)
        E.to_parquet(out / f"candidates_{v}.parquet", index=False)
        with open(out / f"candidates_{v}.jsonl", "w", encoding="utf-8") as f:
            for rec in E.to_dict("records"):
                f.write(json.dumps(_clean(rec), sort_keys=True) + "\n")
    config = dict(method=M, title=METHODS[M]["title"], dataset_pass_column=col, window=cfg.as_dict(),
                  thresholds=dict(taus_s=list(cfg.taus_s) if M in ("A", "C") else None, radius_m=cfg.radius_m if M in ("B", "C") else None,
                                  radius_diagnostic_m=list(cfg.radius_diagnostic_m) if M == "B" else None),
                  combination="AND" if M == "C" else None,
                  model_visible_inputs=["x_m", "y_m", "t_rel_s", "missingness_mask"],
                  metadata_never_model_input=["slot_a", "slot_b", "player_id_a", "player_id_b", "jersey_a", "jersey_b", "side_a", "side_b", "team_relation",
                                              "status", "tier", "quality flags", "event/encounter ids"])
    (out / "config.json").write_text(dumps(config), encoding="utf-8")
    (out / "counts.json").write_text(dumps(counts), encoding="utf-8")
    (out / "rejections.json").write_text(dumps(rej), encoding="utf-8")
    (out / "linear_vs_pchip.json").write_text(dumps(comp), encoding="utf-8")
    return dict(counts=counts, rejections=rej, comparison=comp)


def finish_manifest(out: Path, extra: dict[str, Any]) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    man = dict(extra, outputs_sha256={str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    (out / "manifest.json").write_text(dumps(man), encoding="utf-8")


# ---------------------------------------------------------------- figures

def pick(df: pd.DataFrame, n: int = 1, used: set | None = None) -> list[dict]:
    """First ``n`` events by event id, skipping events already shown (deterministic)."""
    if used is not None:
        df = df[~df["event_id"].isin(used)]
    recs = df.sort_values("event_id").head(n).to_dict("records")
    if used is not None:
        used.update(r["event_id"] for r in recs)
    return recs


def make_figures(M: str, fig_dir: Path, ex_match: str, evs: dict[str, pd.DataFrame], cfg: W.WindowConfig, cache: dict) -> list[dict]:
    from . import figures as F
    fig_dir.mkdir(parents=True, exist_ok=True)
    c, mask, seg, coefs = cache["c"], cache["mask"], cache["seg"], cache["coefs"]
    L = evs["linear"][evs["linear"]["match_id"] == int(ex_match)]
    P = evs["pchip"][evs["pchip"]["match_id"] == int(ex_match)].set_index("event_id")
    col = pass_col(M, cfg)
    tmax = max(cfg.taus_s)
    items: list[tuple[str, dict, str, str, int | None]] = []
    used: set[str] = set()
    passing = L[L[col]]
    for e in pick(passing[(passing["status"] == "accepted") & (passing["tier"] == "T1_visible_visible")], 2, used):
        items.append(("accepted", e, "accepted example (passes the method, quality accepted, both VISIBLE at the intersection)", "", None))
    for e in pick(passing[(passing["status"] == "accepted") & (passing["tier"] != "T1_visible_visible")], 1, used):
        items.append(("accepted_estimated", e, "passing example with ESTIMATED samples at the intersection (kept, tier T2/T3)", "", None))
    for e in pick(passing[passing["status"] != "accepted"].sort_values("status", ascending=False), 1, used):
        items.append(("quality_" + e["status"], e, f"passes the method condition but quality status = {e['status']}", "", None))
    if M == "A":
        fails = L[(L["abs_dt_s"] > tmax) & (L["abs_dt_s"] <= 1.0)]
        for e in pick(fails, 1, used):
            items.append(("rejected_threshold", e, f"rejected: geometric intersection exists but |dt| > {tmax:.2f} s", "", None))
    if M in ("B", "C"):
        fails = L[~L["b_pass"] & (L["anchor_dist_min_m"] <= 1.0)]
        for e in pick(fails, 1, used):
            items.append(("rejected_anchor_distance", e, f"rejected: anchor-time distance > {cfg.radius_m:.2f} m in every window", "", None))
    if M == "C":
        for e in pick(L[L["b_pass"] & ~L[col]], 1, used):
            items.append(("rejected_time_passes_distance", e, f"rejected by C: passes R but |dt| > {tmax:.2f} s (AND not satisfied)", "", None))
    out = []
    for tag, e, why, _x, _a in items:
        anchor = None
        if M in ("B", "C") and json.loads(e["b_pass_windows"]):
            anchor = json.loads(e["b_pass_windows"])[0]
        cond = condition_text(M, e, cfg)
        png = F.event_figure(c, mask, seg, coefs, e, cfg, f"{METHODS[M]['title']}\nmatch {ex_match}, period {e['period']}: {why}", cond, anchor=anchor)
        name = f"{M.lower()}_{tag}_{e['event_id']}.png"
        (fig_dir / name).write_bytes(png)
        out.append(dict(file=name, caption=why, event_id=e["event_id"]))
    # linear vs PCHIP: the passing event with the largest location change (if any matched)
    m_ = passing[passing["pchip_match"].notna()]
    if len(m_):
        e = m_.sort_values(["pchip_loc_diff_m", "event_id"], ascending=[False, True]).iloc[0].to_dict()
        pe = P.loc[e["pchip_match"]].to_dict()
        png = F.event_figure(c, mask, seg, coefs, e, cfg, f"{METHODS[M]['title']}\nlinear (solid) vs PCHIP (dashed), largest location change among passing events",
                             condition_text(M, e, cfg) + f"\nPCHIP: intersection moved {e['pchip_loc_diff_m'] * 100:.1f} cm; |dt| {e['abs_dt_s']:.3f} -> {pe['abs_dt_s']:.3f} s; "
                             f"PCHIP passes = {bool(pe[col])}", show_pchip=True, pchip_ev=pe)
        name = f"{M.lower()}_linear_vs_pchip_{e['event_id']}.png"
        (fig_dir / name).write_bytes(png)
        out.append(dict(file=name, caption="linear vs PCHIP (largest location change)", event_id=e["event_id"]))
    # duplicate windows -> one event
    if len(passing):
        e = passing.sort_values(["n_windows", "event_id"], ascending=[False, True]).iloc[0].to_dict()
        png = F.dedup_figure(e, cfg, f"Method {M}: {e['n_windows']} overlapping windows select this intersection as their first -> one event {e['event_id']}")
        name = f"{M.lower()}_dedup_{e['event_id']}.png"
        (fig_dir / name).write_bytes(png)
        out.append(dict(file=name, caption="overlapping windows merged into one event", event_id=e["event_id"]))
    return out


def condition_text(M: str, e: dict, cfg: W.WindowConfig) -> str:
    taus = ", ".join(f"{t:.2f}:{'pass' if e[f'a_tau_{cfg.tau_key(t)}'] else 'fail'}" for t in cfg.taus_s)
    ad = json.loads(e["anchor_dists_m"])
    b = f"anchor distances per window {[None if x is None else round(x, 3) for x in ad]} m; B (R={cfg.radius_m:.2f}) {'pass' if e['b_pass'] else 'fail'}"
    a = f"|dt| = {e['abs_dt_s']:.3f} s; A per tau [{taus}]"
    q = (f"status {e['status']} {json.loads(e['reject_reasons']) + json.loads(e['flag_reasons'])}; tier {e['tier']}; context before/after "
         f"A {e['before_s_a']:.1f}/{e['after_s_a']:.1f} s, B {e['before_s_b']:.1f}/{e['after_s_b']:.1f} s; "
         f"{e['max_intersections_in_window']} intersection(s) in window; windows {json.loads(e['windows'])}")
    head = {"A": a, "B": b, "C": f"{a}\n{b}\nC = A AND B: {'pass' if e[f'c_tau_{cfg.tau_key(max(cfg.taus_s))}'] else 'fail'} (tau {max(cfg.taus_s):.2f})"}[M]
    return f"{head}\n{q}\nwindow min raw distance (diagnostic only) {e['window_min_raw_dist_m']:.3f} m"


def comparison_figures(fig_dir: Path, results: dict, evs: dict[str, pd.DataFrame], cfg: W.WindowConfig) -> list[dict]:
    from . import figures as F
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []

    def save(name, png, cap):
        (fig_dir / name).write_bytes(png)
        out.append(dict(file=name, caption=cap))

    labels, det_, ev_, enc_ = [], [], [], []
    for M in METHODS:
        for lab, r in results[M]["counts"]["linear"].items():
            labels.append(f"{M}\n{lab.replace(' & ', chr(10))}")
            det_.append(r["raw_pair_window_detections"]), ev_.append(r["independent_events"]), enc_.append(r["distinct_encounters"])
    save("counts_by_method.png", F.bars(labels, {"raw pair-window detections": det_, "deduplicated events": ev_, "distinct encounters": enc_},
                                        "Candidate counts by method and threshold (linear)", "count (log scale)", log=True), "candidate counts")
    main = {M: list(results[M]["counts"]["linear"].values())[-1] for M in METHODS}
    groups = [f"{M} ({list(results[M]['counts']['linear'])[-1]})" for M in METHODS]
    tiers = ["T1_visible_visible", "T2_mixed", "T3_estimated_estimated"]
    save("quality_tiers.png", F.bars(groups, {t: [main[M]["tier"].get(t, 0) for M in METHODS] for t in tiers}, "Quality tier at the intersection (linear, loosest threshold)",
                                     "events", stacked=True), "quality tiers")
    sts = ["accepted", "flagged", "rejected"]
    save("status.png", F.bars(groups, {s: [main[M]["status"].get(s, 0) for M in METHODS] for s in sts}, "Validation status (linear, loosest threshold)", "events",
                              stacked=True), "validation status")
    reasons = sorted({r for M in METHODS for r in list(main[M]["reject_reasons"]) + list(main[M]["flag_reasons"])})
    if reasons:
        save("rejection_reasons.png", F.bars([r.replace("_in_context", "").replace("_", " ") for r in reasons],
                                             {M: [main[M]["reject_reasons"].get(r, 0) + main[M]["flag_reasons"].get(r, 0) for r in reasons] for M in METHODS},
                                             "Reject (corruption) and flag (suspected artifact) reasons, events with the reason for either player", "events",
                                             note="duplicate position / unattributed -> rejected; jump, frame-wide shift, discontinuity, substitution -> flagged"), "rejection and flag reasons")
    L, P = evs["linear"], evs["pchip"]
    sens = {}
    for v, E in (("linear", L), ("pchip", P)):
        sens[v] = [int(E[f"a_tau_{cfg.tau_key(t)}"].sum()) for t in cfg.taus_s] + [int(E["b_pass"].sum())] + \
                  [int(E[f"b_diag_{cfg.tau_key(r)}"].sum()) for r in cfg.radius_diagnostic_m] + [int(E[f"c_tau_{cfg.tau_key(t)}"].sum()) for t in cfg.taus_s]
    sl = [f"A tau {t:.2f}" for t in cfg.taus_s] + [f"B R {cfg.radius_m:.2f}"] + [f"B R {r:.2f}\n(diagnostic)" for r in cfg.radius_diagnostic_m] + \
         [f"C tau {t:.2f}" for t in cfg.taus_s]
    save("threshold_sensitivity.png", F.bars(sl, sens, "Threshold sensitivity: deduplicated events (linear vs PCHIP)", "events", log=True,
                                             note="B at 0.50 and 1.00 m is a diagnostic sensitivity only; no dataset is built from it"), "threshold sensitivity")
    tk = cfg.tau_key(max(cfg.taus_s))
    ov = {v: [int((E[f"a_tau_{tk}"] & ~E["b_pass"]).sum()), int((~E[f"a_tau_{tk}"] & E["b_pass"]).sum()), int((E[f"a_tau_{tk}"] & E["b_pass"]).sum())]
          for v, E in (("linear", L), ("pchip", P))}
    save("dataset_overlap.png", F.bars(["A only", "B only", "A and B (= C)"], ov, f"Dataset overlap (tau {max(cfg.taus_s):.2f} s, R {cfg.radius_m:.2f} m)", "events"),
         "dataset overlap")
    ctx = {k: [main[M]["context"][k] / max(1, main[M]["independent_events"]) for M in METHODS] for k in ("context_full_2_5s_both", "context_min_1s_both", "sample_ready")}
    save("context_completeness.png", F.bars(groups, ctx, "Context completeness (share of events, linear)", "share of events",
                                            note="full: both players have 2.5 s of unbroken valid trajectory before and after the event time; "
                                                 "sample_ready: >= 1 s both sides and status accepted"), "context completeness")
    mt = L["pchip_match"].notna()
    save("pchip_location_change.png", F.hist({"all primary events (linear matched to PCHIP)": L.loc[mt, "pchip_loc_diff_m"] * 100},
                                             "Linear vs PCHIP: intersection location change", "cm", np.linspace(0, 20, 41),
                                             note=f"matched {int(mt.sum()):,} of {len(L):,} linear events; unmatched linear {int((~mt).sum()):,}, PCHIP-only {int(P['linear_match'].isna().sum()):,}"),
         "PCHIP location change")
    save("pchip_dt_change.png", F.hist({"|dt| change (PCHIP - linear)": L.loc[mt, "pchip_abs_dt_change_s"] * 1000}, "Linear vs PCHIP: arrival-time difference change",
                                       "ms", np.linspace(-50, 50, 51)), "PCHIP dt change")
    return out


def write_index(path: Path, sections: list[tuple[str, Path, list[dict]]], intro: str) -> None:
    rel = lambda p: Path(*[".."] * 0, p).as_posix()  # noqa: E731
    html = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Window Intersection Figures</title><style>body{font-family:system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b;max-width:1200px}"
            "figure{margin:12px 0 28px}img{max-width:100%;border:1px solid #e4e3df}figcaption{color:#52514e;font-size:14px}"
            "@media (prefers-color-scheme: dark){body{background:#1a1a19;color:#f0efec}figcaption{color:#c3c2b7}}</style></head><body>",
            f"<h1>M11 window-intersection visual inspection</h1><p>{intro}</p>"]
    md = ["# M11 window-intersection visual inspection", "", intro, ""]
    for title, base, figs in sections:
        html.append(f"<h2>{title}</h2>")
        md += [f"## {title}", ""]
        for f in figs:
            p = Path(__import__("os").path.relpath(base / f["file"], path.parent)).as_posix()
            html.append(f"<figure><img src='{p}' alt='{f['caption']}'><figcaption>{f['caption']} &mdash; <code>{p}</code></figcaption></figure>")
            md.append(f"- [{f['caption']}]({p})")
        md.append("")
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["pilot", "full"], required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--brute", type=int, default=None, help="brute-force pair-windows per match and variant (default: 400 pilot, 0 full)")
    ap.add_argument("--example-match", default=EXAMPLE_MATCH_DEFAULT)
    a = ap.parse_args(argv)
    devs = dev_matches()
    matches = a.matches or devs
    bad = [m for m in matches if m not in devs]
    if bad:
        raise SystemExit(f"not development matches (refused): {bad}")
    if a.example_match not in matches:
        a.example_match = matches[0]
    brute_n = a.brute if a.brute is not None else (400 if a.layout == "pilot" else 0)
    shared, mdirs = layout_dirs(a.layout)
    for d in [shared, *mdirs.values()]:
        if d.exists():
            raise SystemExit(f"{d} exists; versioned outputs are never overwritten")
    cfg = W.WindowConfig()
    snap0 = validate.raw_snapshot(RAW)
    t0 = time.perf_counter()
    shared.mkdir(parents=True)
    jobs = [(m, str(shared), cfg.as_dict(), brute_n) for m in matches]
    results = []
    def report(r):
        print(f"{r['match']}: {r['summary']['seconds']} s, events linear {r['summary']['scan']['linear']['events']}", flush=True)
        results.append(r)

    if a.workers <= 1:
        for j in jobs:
            report(process_match(j))
    else:
        with Pool(a.workers) as pool:
            for r in pool.imap(process_match, jobs):
                report(r)
    evs = {v: pd.concat([r["events"][v] for r in results], ignore_index=True) for v in W.VARIANTS}
    dets = {v: pd.concat([r["detections"][v] for r in results], ignore_index=True) for v in W.VARIANTS}
    code = code_sha256()
    # shared summary
    tot = {v: {} for v in W.VARIANTS}
    for r in results:
        for v in W.VARIANTS:
            sc = r["summary"]["scan"][v]
            for k, x in list(sc["window"].items()) + list(sc["hits"].items()) + [(k, sc[k]) for k in ("pair_window_detections", "pair_windows_multi_intersection",
                                                                                                     "events", "intersections_never_primary")]:
                tot[v][k] = tot[v].get(k, 0) + x
    seg_tot: dict[str, int] = {}
    for r in results:
        for k, x in r["summary"]["segments"].items():
            seg_tot[k] = seg_tot.get(k, 0) + x
    checks = {r["match"]: r["summary"]["checks"] for r in results}
    all_ok = all(all(v for k, v in ch[var].items() if isinstance(v, bool)) for ch in checks.values() for var in W.VARIANTS)
    brute_mism = sum(ch.get("brute_force", {}).get(v, {}).get("mismatches", 0) for ch in checks.values() for v in W.VARIANTS)
    pchip_exc = max(ch["pchip"]["max_excursion_outside_sample_bbox_m"] for ch in checks.values())
    shared_summary = dict(schema=W.SCHEMA, layout=a.layout, matches=matches, config=cfg.as_dict(), totals=tot, segments=seg_tot,
                          linear_vs_pchip={r["match"]: r["summary"]["linear_vs_pchip"] for r in results},
                          checks=dict(all_invariants_pass=all_ok, brute_force_mismatches=brute_mism, pchip_max_excursion_m=pchip_exc, per_match=checks),
                          seconds_extraction=round(time.perf_counter() - t0, 1))
    (shared / "summary.json").write_text(dumps(shared_summary), encoding="utf-8")
    # method datasets
    mres = {M: write_method(M, mdirs[M], evs, dets, cfg, matches, code, a.layout) for M in METHODS}
    # figures
    from . import figures as F  # noqa: F401
    c, mask, _e, _m, _p = load_source(str(RAW), a.example_match)
    seg, _cnt, _dup = W.build_segments(c, mask, load_steps(str(RAW), a.example_match))
    coefs = {v: W.coefficients(seg, v)[:2] for v in W.VARIANTS}
    cache = dict(c=c, mask=mask, seg=seg, coefs=coefs)
    sections = []
    for M in METHODS:
        figs = make_figures(M, mdirs[M] / "figures", a.example_match, evs, cfg, cache)
        sections.append((METHODS[M]["title"], mdirs[M] / "figures", figs))
        (mdirs[M] / "figures" / "figures.json").write_text(dumps(figs), encoding="utf-8")
    comp = comparison_figures(shared / "figures", mres, evs, cfg)
    sections.insert(0, ("Method-level comparison", shared / "figures", comp))
    write_index(shared / "visual_index.html", sections,
                f"Actual FIFA development data ({len(matches)} match(es)); example events from match {a.example_match}. Layout: {a.layout}.")
    snap1 = validate.raw_snapshot(RAW)
    common = dict(schema=W.SCHEMA, layout=a.layout, matches=matches, split="development only", config=cfg.as_dict(), code_sha256=code,
                  raw_unchanged=snap0 == snap1, raw_snapshot=snap1, created_unix=int(time.time()), argv=sys.argv[1:])
    for M in METHODS:
        finish_manifest(mdirs[M], dict(common, method=M))
    finish_manifest(shared, common)
    print(dumps(dict(all_invariants_pass=all_ok, brute_force_mismatches=brute_mism, pchip_max_excursion_m=pchip_exc, raw_unchanged=snap0 == snap1,
                     totals_linear=tot["linear"], methods={M: {v: {k: x["independent_events"] for k, x in mres[M]["counts"][v].items()} for v in W.VARIANTS}
                                                         for M in METHODS})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
