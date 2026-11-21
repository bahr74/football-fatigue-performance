"""Build Interaction Catalogue v1 on the 32 M11 train/CV development matches (CPU only, no model, no training).

    python -m football_fatigue.interaction_catalogue run --layout pilot --matches 3812 3816 --workers 2
    python -m football_fatigue.interaction_catalogue run --layout full --workers 4
    python -m football_fatigue.interaction_catalogue summarize --layout full

Protected data: only matches in ``train_cv`` of the frozen ``m11-stage2-split-v1/split.json`` are allowed. The internal held-out
matches, every non-development match of ``m11-encounters-v1/splits.json`` (official validation / test) and any match not in
train_cv are refused *before* anything is loaded. Outputs go to new versioned directories; an existing match directory is never
overwritten. RAW data and all earlier M11 outputs are read only.
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

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import QFLAG_BIT, STATE_AVAILABLE
from football_fatigue.ingestion import Dataset
from football_fatigue.ingestion.models import QualityFlag
from football_fatigue.intersections.catalogue import load_steps
from football_fatigue.matching.benchmark import load_source
from football_fatigue.window_intersections import core as W

from . import core as K

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SPLIT = PROC / "m11-stage2-split-v1" / "split.json"
ENC_SPLITS = PROC / "m11-encounters-v1" / "splits.json"
LAYOUTS = {"full": "m11-interaction-catalogue-v1", "pilot": "m11-interaction-catalogue-pilot-v1"}
TABLES = ("nodes", "edges", "edge_chains", "crossings", "pair_windows", "groups", "group_timeline", "window_stats")


# ---------------------------------------------------------------- protected-data boundary

def allowed_matches(split_path: Path = SPLIT, enc_path: Path = ENC_SPLITS) -> list[str]:
    """The 32 train/CV matches. Fails if the frozen split is inconsistent with the protected sets."""
    s = json.loads(split_path.read_text(encoding="utf-8"))
    e = json.loads(enc_path.read_text(encoding="utf-8"))
    train = sorted((str(m) for m in s["train_cv"]), key=int)
    protected = protected_matches(split_path, enc_path)
    if len(train) != 32 or set(train) & protected:
        raise SystemExit("frozen split inconsistent: train_cv must hold 32 matches disjoint from every protected match")
    if any(e["matches"].get(m, {}).get("split") != "development" for m in train):
        raise SystemExit("a train_cv match is not a development match")
    return train


def protected_matches(split_path: Path = SPLIT, enc_path: Path = ENC_SPLITS) -> set[str]:
    s = json.loads(split_path.read_text(encoding="utf-8"))
    e = json.loads(enc_path.read_text(encoding="utf-8"))
    out = {str(m) for m in s["heldout"]} | {str(m) for m in s["approved_heldout"]}
    out |= {str(m) for m, v in e["matches"].items() if v["split"] != "development"}
    return out


def assert_allowed(m: str, allowed: list[str]) -> None:
    if str(m) not in set(allowed):
        raise SystemExit(f"match {m} is not an approved train/CV development match; refused (nothing was loaded)")


# ---------------------------------------------------------------- helpers

def code_sha256() -> dict[str, str]:
    base = Path(__file__).parents[1]
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [base / "window_intersections" / n for n in ("core.py", "candidates.py", "local.py")]
    files += [base / "intersections" / n for n in ("exact.py", "detect.py", "catalogue.py")]
    files += [base / "encounters" / "quality.py", base / "matching" / "benchmark.py"]
    return {str(p.relative_to(base)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def dumps(o: Any) -> str:
    return json.dumps(_clean(o), indent=2, sort_keys=True)


def coordinate_checks(c, pitch_len: float, pitch_wid: float, margin: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Out-of-pitch mask under the centred-origin convention, after checking the observed coordinate range supports it."""
    av = c.state == STATE_AVAILABLE
    x, y = c.x[av], c.y[av]
    q = dict(x_p001=float(np.quantile(x, 0.001)), x_p999=float(np.quantile(x, 0.999)), y_p001=float(np.quantile(y, 0.001)),
             y_p999=float(np.quantile(y, 0.999)), x_min=float(x.min()), x_max=float(x.max()), y_min=float(y.min()), y_max=float(y.max()))
    centred = (q["x_p001"] < 0 < q["x_p999"]) and (q["y_p001"] < 0 < q["y_p999"]) and \
        max(abs(q["x_p001"]), q["x_p999"]) <= pitch_len / 2 + 10 and max(abs(q["y_p001"]), q["y_p999"]) <= pitch_wid / 2 + 10
    oob = np.zeros(c.x.shape, bool)
    if centred:
        with np.errstate(invalid="ignore"):
            oob = av & ((np.abs(c.x) > pitch_len / 2 + margin) | (np.abs(c.y) > pitch_wid / 2 + margin))
    inv_bit = QFLAG_BIT[QualityFlag.INVALID_COORDINATE]
    return oob, dict(pitch_length_m=pitch_len, pitch_width_m=pitch_wid, centred_origin_supported=bool(centred), observed=q,
                     oob_margin_m=margin, oob_samples=int(oob.sum()), available_samples=int(av.sum()),
                     available_non_finite=int((av & ~(np.isfinite(c.x) & np.isfinite(c.y))).sum()),
                     invalid_coordinate_flagged_samples=int(((c.qflags & inv_bit) != 0).sum()))


def stage2_check(m: str, chains: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    """At R = 0.5 m and a 1.0 s merge gap the catalogue must reproduce the historical Stage 2 episodes and candidates exactly."""
    p = PROC / "m11-stage2-candidates-v1" / "matches" / m / "candidates.parquet"
    if not p.exists():
        return dict(available=False)
    s2 = pd.read_parquet(p, columns=["period", "slot_a", "slot_b", "t_star", "d_min", "n_episodes"])
    ch = chains[chains["radius_m"] == 0.5]
    a = s2.sort_values(["period", "slot_a", "slot_b", "t_star"]).reset_index(drop=True)
    b = ch.sort_values(["period", "slot_a", "slot_b", "t_star"]).reset_index(drop=True)
    same_n = len(a) == len(b)
    keys = same_n and bool((a[["period", "slot_a", "slot_b"]].to_numpy() == b[["period", "slot_a", "slot_b"]].to_numpy()).all())
    t_ok = same_n and bool(np.allclose(a["t_star"], b["t_star"], atol=1e-6) and np.allclose(a["d_min"], b["d_min"], atol=1e-9))
    ne = same_n and bool((a["n_episodes"].to_numpy() == b["n_edges"].to_numpy()).all())
    return dict(available=True, stage2_candidates=len(a), catalogue_chains=len(b), stage2_episodes=int(s2["n_episodes"].sum()),
                catalogue_edges=int((edges["radius_m"] == 0.5).sum()), keys_equal=keys, t_star_and_d_min_equal=t_ok, n_edges_equal=ne,
                passed=bool(same_n and keys and t_ok and ne and int(s2["n_episodes"].sum()) == int((edges["radius_m"] == 0.5).sum())))


def v1_check(m: str, cr: pd.DataFrame, pw: pd.DataFrame) -> dict[str, Any]:
    p = PROC / "m11-window-intersections-v1" / m / "match_summary.json"
    if not p.exists():
        return dict(available=False)
    s = json.loads(p.read_text(encoding="utf-8"))["scan"]["linear"]
    r = dict(available=True, v1_intersections=s["hits"]["intersections"], catalogue_crossings=len(cr),
             v1_pair_window_detections=s["pair_window_detections"], catalogue_pair_windows=len(pw),
             v1_multi=s["pair_windows_multi_intersection"], catalogue_multi=int((pw["n_intersections"] > 1).sum()),
             v1_events_primary=s["events"], v1_never_primary=s["intersections_never_primary"])
    r["passed"] = (r["v1_intersections"] == r["catalogue_crossings"] and r["v1_pair_window_detections"] == r["catalogue_pair_windows"]
                   and r["v1_multi"] == r["catalogue_multi"])
    return r


def structural_checks(t: dict[str, pd.DataFrame], cfg: K.CatalogueConfig) -> dict[str, Any]:
    e = t["edges"]
    out = dict(edge_ids_unique=bool(e["edge_id"].is_unique), slot_order=bool((e["slot_a"] < e["slot_b"]).all()),
               t_star_inside=bool(((e["t_star"] >= e["start"] - 1e-9) & (e["t_star"] <= e["end"] + 1e-9)).all()),
               d_min_le_radius=bool((e["d_min"] <= e["radius_m"] + 1e-9).all()),
               every_edge_one_group=bool(all(e[f"group_d{d:.2f}".replace(".", "p")].notna().all() for d in cfg.group_link_s)))
    g = t["groups"]
    out["group_edges_partition"] = bool(all(
        int(g[(g["radius_m"] == R) & (g["link_s"] == d)]["n_edges"].sum()) == int((e["radius_m"] == R).sum())
        for R in cfg.radii_m for d in cfg.group_link_s))
    # nesting: every edge at a smaller radius lies inside an edge of the same chain pair at the next radius
    nest = True
    radii = sorted(cfg.radii_m)
    for r0, r1 in zip(radii[:-1], radii[1:]):
        a = e[e["radius_m"] == r0]
        b = e[e["radius_m"] == r1]
        bb = {k: g for k, g in b.groupby(["period", "chain_a", "chain_b"])}
        for k, ga in a.groupby(["period", "chain_a", "chain_b"]):
            gb = bb.get(k)
            if gb is None:
                nest = False
                break
            s, en = gb["start"].to_numpy(), gb["end"].to_numpy()
            inside = ((ga["start"].to_numpy()[:, None] >= s[None] - 1e-9) & (ga["end"].to_numpy()[:, None] <= en[None] + 1e-9)).any(1)
            if not inside.all():
                nest = False
                break
    out["radii_nested"] = nest
    out["passed"] = all(v for v in out.values())
    return out


# ---------------------------------------------------------------- one match

def process_match(args: tuple) -> dict[str, Any]:
    m, out_root, cfg_d, allowed = args
    assert_allowed(m, allowed)                    # before anything is loaded
    cfg = K.CatalogueConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_d.items()})
    t0 = time.perf_counter()
    c, mask, _enc, src_manifest, prov = load_source(str(RAW), m)
    if str(c.match_id) != m:
        raise SystemExit(f"loaded cache is match {c.match_id}, expected {m}")
    md = Dataset(str(RAW)).metadata(m)
    steps = load_steps(str(RAW), m)
    seg, seg_counts, dup = W.build_segments(c, mask, steps)
    ch = W.chain_table(seg)
    cx, cy, _ = W.coefficients(seg, "linear")
    oob, coord = coordinate_checks(c, md.pitch_length_m, md.pitch_width_m, cfg.oob_margin_m)
    av = c.state == STATE_AVAILABLE
    ts_checks, parts = {}, {k: [] for k in TABLES}
    for p in sorted(int(v) for v in np.unique(seg["period"])):
        rows = np.flatnonzero(c.canonical_period == p)
        tp = c.video_time_s[rows]
        dt = np.diff(tp)
        ts_checks[p] = dict(rows=len(rows), start_s=float(tp[0]), end_s=float(tp[-1]), non_increasing_steps=int((dt <= 0).sum()),
                            non_finite=int((~np.isfinite(tp)).sum()), frame_gaps=int((c.frame_gap_before[rows] > 0).sum()),
                            max_step_s=float(dt.max()) if len(dt) else 0.0)
        if ts_checks[p]["non_increasing_steps"] or ts_checks[p]["non_finite"]:
            raise SystemExit(f"match {m} period {p}: timestamps not strictly increasing; catalogue not built")
        fc = K.FrameContext(t=tp, mask=mask[rows], avail=av[rows], dup=dup[rows], oob=oob[rows])
        res = K.period_catalogue(seg, ch, cx, cy, p, float(tp[0]), float(tp[-1]), fc, c.slot_side, int(m), cfg)
        for k, v in res.items():
            if len(v):
                parts[k].append(v)
    parts["nodes"].append(K.nodes_table(seg, ch, c.slot_player_id, c.slot_side, int(m)))
    t = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in parts.items()}
    e = t["edges"]
    e["player_id_a"], e["player_id_b"] = c.slot_player_id[e["slot_a"]], c.slot_player_id[e["slot_b"]]
    checks = dict(stage2_reproduction=stage2_check(m, t["edge_chains"], e), window_intersections_v1_reproduction=v1_check(m, t["crossings"], t["pair_windows"]),
                  structure=structural_checks(t, cfg))
    sample_quality = {k: int(((mask & bit) != 0).sum()) for k, bit in (
        ("estimated", Q.ESTIMATED), ("update_instants", Q.UPDATE_INSTANT), ("implausible_jump", Q.IMPLAUSIBLE_JUMP),
        ("overspeed_step", Q.OVERSPEED_STEP), ("frame_wide_shift", Q.FRAME_WIDE_SHIFT), ("discontinuity_candidate", Q.DISCONTINUITY_CANDIDATE),
        ("label_transfer_candidate", Q.LABEL_TRANSFER_CANDIDATE), ("reciprocal_exchange", Q.RECIPROCAL_EXCHANGE),
        ("substitution_window", Q.SUBSTITUTION_WINDOW))}
    sample_quality["duplicate_position"] = int(dup.sum())
    out = Path(out_root) / "matches" / m
    out.mkdir(parents=True, exist_ok=False)
    for k, v in t.items():
        v.to_parquet(out / f"{k}.parquet", index=False)
    summary = dict(schema=K.SCHEMA, match_id=int(m), split="train_cv", segments=seg_counts, timestamps=ts_checks, coordinates=coord,
                   sample_quality=sample_quality, counts={k: len(v) for k, v in t.items()}, checks=checks,
                   source=dict(step1=prov, step1_schema=src_manifest["schema"]), seconds=round(time.perf_counter() - t0, 1))
    (out / "match_summary.json").write_text(dumps(summary), encoding="utf-8")
    return dict(match=m, seconds=summary["seconds"], passed=bool(checks["structure"]["passed"]
                and checks["stage2_reproduction"].get("passed", True) and checks["window_intersections_v1_reproduction"].get("passed", True)))


# ---------------------------------------------------------------- run / summarize

def run(args: argparse.Namespace) -> int:
    allowed = allowed_matches()
    matches = [str(m) for m in (args.matches or allowed)]
    for m in matches:
        assert_allowed(m, allowed)
    out_root = PROC / LAYOUTS[args.layout]
    for m in matches:
        if (out_root / "matches" / m).exists():
            raise SystemExit(f"{out_root / 'matches' / m} exists; never overwritten")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = K.CatalogueConfig()
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    man = dict(schema=K.SCHEMA, layout=args.layout, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), matches=matches, config=cfg.as_dict(),
               code_sha256=code_sha256(), split_file=str(SPLIT.relative_to(ROOT)), split_sha256=hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
               protected_matches_refused=sorted(protected_matches(), key=int), python=sys.version, platform=platform.platform(),
               note="Descriptive catalogue. No model, no training, no held-out or official validation/test access.")
    t0 = time.perf_counter()
    jobs = [(m, str(out_root), cfg.as_dict(), allowed) for m in matches]
    res = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(process_match, jobs):
                print(r, flush=True)
                res.append(r)
    else:
        for j in jobs:
            r = process_match(j)
            print(r, flush=True)
            res.append(r)
    man.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.perf_counter() - t0, 1), results=sorted(res, key=lambda r: int(r["match"])))
    (out_root / f"run_manifest_{stamp}.json").write_text(dumps(man), encoding="utf-8")
    return 0 if all(r["passed"] for r in res) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.interaction_catalogue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--layout", choices=sorted(LAYOUTS), default="pilot")
    r.add_argument("--matches", nargs="*")
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--layout", choices=sorted(LAYOUTS), default="full")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    from .summary import summarize
    return summarize(PROC / LAYOUTS[a.layout], allowed_matches())


if __name__ == "__main__":
    raise SystemExit(main())
